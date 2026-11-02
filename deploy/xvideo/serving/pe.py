import asyncio
import base64
import logging
import os
import re
from io import BytesIO
from typing import List, Optional

from PIL import Image
import httpx

logger = logging.getLogger("joyomni.pe")

DEFAULT_MODEL = os.environ.get("PE_MODEL", "")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MAX_RETRIES = 8


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


PE_IMAGE_MAX_SIDE = _env_int("PE_IMAGE_MAX_SIDE", 768)

SYSTEM_PROMPT = """You write precise English instructions for video-to-video editing. Use the source
frames to interpret the user's requested changes. Preserve the user's intent and explicit
exceptions, and return only the final editing prompt."""

V2V_TEMPLATE = """# INPUT
User request: {user_prompt}
The provided images are SOURCE VIDEO FRAMES, showing the video to edit.

# OUTPUT
Return only one cohesive English paragraph, starting with a direct editing instruction.
Use about 60-120 words for a typical request, fewer for a simple edit. Include every requested
change and exception; completeness takes priority over the word target. Put the main edit first,
add only useful visual details, and end with one brief preservation clause when needed.
No explanations, labels, lists, reasoning, generic quality claims or repeated negative commands.

# SCOPE AND GROUNDING
The user's explicit edits, exclusions and requested strength override every default below.
Identify the combined edit scope before deciding what to preserve. Keep attributes outside that
scope; never preserve an attribute that another requested edit changes.
Use the source frames to identify visible targets, clothing layers, scene layout and framing.
Do not invent source objects, hidden anatomy or occluded details, including in removal instructions.
Newly requested objects, appearances and environments may differ from the source; describe only
details needed to define the requested result. A broad request is not permission to add unrelated
objects or edits.
Keep source action, timing, camera, framing, visual medium and lighting unless the request changes
them. Adjust only local contact, shadows or reflections needed to integrate the edit naturally.

# EDIT TYPES
Use only the rules relevant to the request; combine multiple edits without dropping any of them.

- Add, replace, remove or modify: name the target and the requested result directly. Locate new
  elements using visible scene relationships. Express exclusions as properties of the requested
  result; give removal instructions only for objects visible in the source. For removal, reconstruct
  the exposed region with surrounding textures and lighting. For an attribute edit, change only
  that attribute.

- Facial identity replacement: define the replacement using the target face and any accompanying
  hairstyle or head/face features, excluding attributes the user explicitly retains. Treat all
  transferred features as changed when writing the final preservation clause. Preserve source
  clothing and body unless separately edited; do not import the target's outfit or props.
  Attribute-only edits, such as aging or hair color, change only the requested attributes.

- Whole-person, character or creature replacement: describe recognizable head/face features and
  the visible body or outfit that define the requested replacement. A costume change alone is not
  an identity replacement. Do not preserve source identity, clothing or anatomy that the requested
  replacement changes, or invent extra props. Honor explicitly retained clothing and accessories.
  For a material transformation, express retained likeness through that material, rather than
  claiming the original skin or hair remains unchanged.

- Clothing and footwear: change the requested item or layer, preserving other garments,
  accessories and source body proportions. Fit the new garment naturally without imposing a
  slender silhouette or changing the requested loose/tailored cut. A color-only edit retains cut.
  Trouser-only edits retain waistband height and the top's hem, drape and overlap unless requested
  otherwise. Describe requested sleeve length or trouser flare as a garment construction and
  silhouette change, not an extra layer; match the requested cut and degree of flare.
  For a qipao/cheongsam, describe its requested collar, fastenings, fabric and cut, with connected
  panels deforming naturally. Preserve the requested slit design; default its upper endpoint to
  near knee level unless another height is specified. Keep that endpoint fixed on the garment
  during motion, with stable panel overlap above it. A slit is a side opening, not a raised hem;
  omit offscreen details.
  For an unspecified apricot clothing/footwear shade, use only "warm off-white" throughout the
  output, never "apricot"; an explicit orange/fruit hue, color code or precise shade takes priority.

- Background replacement: describe the requested new environment and fill the replaced regions,
  including gaps around the subject and frame edges. Explicitly remove visible old background
  remnants bordering the subject, such as chair backs or headrests, only when they belong to the
  replacement scope. Name only visible components; never infer a headrest from a chair back.
  Retain surviving foreground elements and objects the user explicitly keeps. Restyling an
  existing background changes its appearance, not its objects or layout, unless also requested.

- Style, weather, lighting, color grading or focus: apply only the requested effect to the requested
  regions and at the requested strength. A whole-frame style change includes the subject and
  background except explicit exclusions, such as retaining a photorealistic face. Preserve
  recognizable identity, action and scene layout within styled regions without claiming their
  original rendering stays unchanged. A style request with "keep everything unchanged" preserves
  scene content and layout while changing the requested appearance. Do not force dramatic effects
  when the user asks for a subtle change.

- Camera or speed: change camera motion, framing, depth of field or action speed only when
  requested. Keep the remaining source motion and scene behavior.

# TEXT
When the user specifies text or a logo to render, preserve the exact supplied string in double
quotes, including its original language, characters and spelling, whether or not the user quoted
it. Do not translate or transliterate it. Do not transcribe, describe or anchor incidental source
text, labels, brands or watermarks unless requested. For text/overlay removal, identify the
requested removal scope without unnecessarily copying its wording.

# VIDEO CONSISTENCY AND FRAMING
Describe the edited result as already present in the first output frame, unless the user requests
a gradual transformation or a timed appearance. Follow the corresponding source frames' motion
rather than freezing the person in the provided frame's pose. Keep coherent tracking, scale,
occlusion and relevant garment deformation, contact, shadows or reflections.
Respect the source crop; do not zoom out, reveal hidden body parts or force a full-body view to
show the edit. A complete outfit request still applies to the entire outfit as it becomes visible
during source motion. Describe currently visible details without inventing offscreen features.
Write the final preservation clause against the edited result: exclude source appearances
replaced, covered or transformed by the edit. Retain unaffected motion and structure without
restoring their old appearance. If the background is unaffected, briefly state that it stays
unchanged; otherwise preserve only its unaffected attributes.
"""

RV2V_SYSTEM_PROMPT = """You write precise English instructions for reference-image-guided video
editing (RV2V). Understand the source video and the separate reference image, then describe only
the requested transfer. The source video determines the subject, pose, motion, framing and scene;
the reference supplies only the visual attributes requested by the user."""

RV2V_TEMPLATE = """# INPUT
User request: {user_prompt}
Image 0 is the REFERENCE IMAGE: the donor of the requested appearance, garment, object or scene.
Any subsequent images are SOURCE VIDEO FRAMES: the video to edit.

# OUTPUT
Return ONLY one cohesive English editing prompt, about 80-140 words. No explanation or labels.
Start with a direct edit instruction, naming the target in the video and the requested item from
the reference image. Refer to it as "the reference image", never by image number in the output.
Inspect the reference and describe 3-5 distinctive visible features relevant to the request:
color, shape, material, pattern and construction. Be concrete and accurate; do not invent details.
Keep the entire request, including any explicit exceptions, extra edits and exact quoted text.

# TRANSFER SCOPE
Explicit requested edits and exceptions override the default preservation rules below. Determine
all requested changes first; preserve only attributes outside that combined edit scope.
- Clothes: identify the visible source clothing layers and the requested replacement scope before
  choosing what to preserve. A request explicitly targeting an inner top, outer garment or trousers
  changes only that layer; a complete outfit transfers the visible outfit. For an unqualified
  request to replace the top/upper-body clothing, make the reference top the visible upper-body
  garment: replace existing layers that would cover or conceal it, including outerwear or
  shoulder-draped coverings, unless the user explicitly keeps them. Do not classify these clothing
  layers as accessories to preserve. Top-only edits keep lower-body clothing, and trousers-only
  edits keep upper-body clothing. Name the source garments/layers being replaced in the final
  prompt, then describe the reference garment's cut, silhouette, length, color and visible details.
  Preserve the source person's identity, face, hair, body proportions and pose within the unchanged
  scope.
  The reference model's face, body, pose and background are not part of a clothing transfer.
  For requested or reference flared trousers, prioritize a fitted knee and seams widening
  below it to loose hems wider than the knee; keep a slight flare modest. Trouser-only edits
  retain the source waistband height and the original top hem, drape and overlap.
  For requested/reference full-length sleeves, prioritize continuous fabric from shoulders
  across elbows and forearms to cuffs touching the hand bases, replacing the source sleeve
  silhouette. Omit the source sleeve length; describe only construction inside the source crop.
  Fit transferred garments naturally to the source anatomy, using one concise slender-tailoring
  clause within the requested/reference cut, including loose styles. Attribute-only edits retain cut.
  For qipao replacement, default to one continuous opaque skirt wrapping both legs to the hem,
  with joined panels deforming together. Describe the reference's collar and dress construction
  instead of qipao/cheongsam or slit/opening terms. Explicit cut or exact-copy requests override.
  Retain other reference details, natural overlap/contact and the source crop; omit offscreen parts.
- Clothing/footwear color: follow the reference unless explicitly recoloring; describe its actual
  color rather than echoing an ambiguous user color label. For default apricot recoloring, use only
  "warm off-white" (near white) throughout the final prompt, never "apricot" or beige/tan/brown.
  Explicit fruit/orange hues and exact color codes override; keep other items in their own colors.
- Background: replace the original environment with the environment in the reference, describing
  its main visible structures. Keep the source foreground person and their clothing. Remove old
  background furniture only when it is actually visible. Do not add people from the reference.
- Person/character replacement: describe the reference face/head AND the requested body/outfit.
  Preserve source pose and motion, without preserving identity attributes that must change.
- Object/accessory: transfer only the requested object, at its functional position and scale.
- Style: apply only the requested visual style from the reference to the specified content.

Close with a concise preservation clause for what stays unchanged. Keep the original background
unless the user requests changing it. Do not copy reference framing, pose, lighting, typography,
logos, watermarks, props or other garments unless requested. Do not describe invisible body parts
or force the camera to reveal the whole garment. Avoid vague quality words and excessive negative
instructions. The edit is already fully present in the first output frame and follows source
motion consistently; do not describe a gradual transformation.

# SOURCE FRAMING — CLOTHING EDITS
First identify the source person's actual visible crop and pose. In a seated chest-up or
head-and-shoulders view, begin the final editing prompt by anchoring the edit to that existing
close-up and seated pose. Describe only the reference garment surfaces that can appear inside
this source crop. For a dress, name the visible upper part of the dress and describe its neckline,
shoulder fabric, sleeves and chest texture. For an outfit, describe only its visible jacket/top
and inner layer. Completely omit descriptions of offscreen garments, waist/hip shaping, skirt
length, trouser legs, hemlines, feet and full-body silhouettes from the final prompt, including
negative mentions of those details. This is a visibility rule, not a change in the user's outfit
choice. Full-body source shots still receive the full requested outfit. At each output frame,
match the person's head, shoulder and arm positions, pose, subject scale and framing to the
corresponding source-video frame; let the clothing follow the source motion with natural
deformation. Do not freeze the person in the provided frame's pose or suppress source movement.
Preserve all visible source accessories that fall outside the requested edit scope. Explicitly
apply any requested accessory removal, replacement or modification; never also describe that
accessory as unchanged. Retained accessories follow the source body's motion
and placement, with the new fabric layered naturally around them. State the actual visible crop
directly; do not output an "if visible" or "if cropped" condition. Never enlarge the visible body
region to show a garment.
"""


def _downscale(image: Image.Image, max_side: int = PE_IMAGE_MAX_SIDE) -> Image.Image:
    if max_side and max_side > 0:
        w, h = image.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / longest
            image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    return image


def _pil_to_b64(image: Image.Image) -> str:
    buf = BytesIO()
    _downscale(image.convert("RGB")).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _img_to_b64(item) -> Optional[str]:
    if item is None:
        return None
    if isinstance(item, tuple):
        item = item[0]
    if isinstance(item, Image.Image):
        return _pil_to_b64(item)
    if isinstance(item, str) and os.path.exists(item):
        return _pil_to_b64(Image.open(item))
    if isinstance(item, str):
        return item
    return None


def _video_frames_to_b64(video) -> List[str]:
    if not video:
        return []
    items = video if isinstance(video, list) else [video]
    out = []
    for item in items:
        b64 = _img_to_b64(item)
        if b64:
            out.append(b64)
    return out


def _message_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", "") or "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "") or ""))
        return "".join(parts)
    return str(content)


def _sanitize_enhanced(text: str, fallback: str) -> str:
    if not text:
        return fallback
    def clean_link(match):
        if match.group() in fallback or match.group(2) in fallback:
            return match.group()
        return "" if match.group().startswith("!") else match.group(1)

    def clean_url(match):
        url = match.group().rstrip(".,;:!?，。；：！？)")
        return match.group() if url in fallback else match.group()[len(url):]

    cleaned = re.sub(r"!?\[([^\]]*)\]\(([^)]*)\)", clean_link, text)
    cleaned = re.sub(r'https?://[^\s<>"\'\[\]（）“”]+', clean_url, cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) < 10:
        logger.warning("PE output empty/too short after sanitizing; using raw prompt")
        return fallback
    if cleaned != text.strip():
        logger.warning("PE output contained URL/link noise; sanitized it")
    return cleaned


def _build_messages(system_prompt: str, user_text: str, images_b64: List[str]):
    content = [{"type": "text", "text": user_text}]
    for i, b64 in enumerate(images_b64):
        content.append({"type": "text", "text": f"\n[Image {i}]:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


class PromptEnhancer:

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        max_retries: int = MAX_RETRIES,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.environ.get("PE_MODEL") or DEFAULT_MODEL
        self.anthropic = "/anthropic" in self.base_url
        self.max_retries = max(1, max_retries)

    def _request(self, system_prompt, user_text, images_b64):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if not self.anthropic:
            return f"{self.base_url.rstrip('/')}/chat/completions", headers, {
                "model": self.model, "max_completion_tokens": 8192,
                "messages": _build_messages(system_prompt, user_text, images_b64),
            }
        content = [{"type": "text", "text": user_text}]
        for i, b64 in enumerate(images_b64):
            content.append({"type": "text", "text": f"\n[Image {i}]:"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64}})
        body = {
            "model": self.model, "max_tokens": 4096, "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }
        headers["anthropic-version"] = "2023-06-01"
        return f"{self.base_url.rstrip('/')}/v1/messages", headers, body

    async def _chat(self, system_prompt, user_text, images_b64, raw_fallback):
        url, headers, body = self._request(system_prompt, user_text, images_b64)
        async with httpx.AsyncClient(timeout=90.0) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=body)
                    response.raise_for_status()
                    data = response.json()
                    content = data["content"] if self.anthropic else data["choices"][0]["message"]["content"]
                    return _sanitize_enhanced(_message_content_to_text(content).strip(), raw_fallback)
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                        exc.response.status_code in (408, 409, 429) or exc.response.status_code >= 500
                    )
                    if not retryable or attempt == self.max_retries:
                        raise
                    logger.warning("PE attempt %d/%d failed: %s", attempt, self.max_retries, type(exc).__name__)
                    await asyncio.sleep(min(attempt, 5))

    async def enhance(self, task_type, user_prompt, video=None, ref_image=None, *, timeout=90.0):
        if not user_prompt or not user_prompt.strip():
            return user_prompt
        video_frames = _video_frames_to_b64(video)
        if task_type == "rv2v":
            reference = _img_to_b64(ref_image)
            if reference is None:
                logger.warning("RV2V PE needs a reference image; using raw prompt")
                return user_prompt
            text = RV2V_TEMPLATE.format(user_prompt=user_prompt)
            system, images = RV2V_SYSTEM_PROMPT, [reference, *video_frames]
        else:
            text = V2V_TEMPLATE.format(user_prompt=user_prompt)
            system, images = SYSTEM_PROMPT, video_frames
        return await asyncio.wait_for(self._chat(system, text, images, user_prompt), timeout=timeout)

    def __call__(self, task_type, user_prompt, video=None, ref_image=None, *, timeout=90.0) -> Optional[str]:
        try:
            return asyncio.run(self.enhance(task_type, user_prompt, video, ref_image, timeout=timeout))
        except Exception as exc:
            logger.warning("PE failed: %s; using raw prompt", type(exc).__name__)
            return user_prompt


MIMO_PERSON_SYSTEM_PROMPT = """You write concise English captions for reference-guided person replacement in a video.
Image 0 is the reference person. Image 1 is a frame of the source video to edit.
Return ONLY one direct editing sentence, about 20-40 words, in this format:
Replace [brief identification of the source person] with the person from image0, featuring [2-3 distinctive visible reference features].
Use reference hair, clothing and accessories to identify the requested appearance. Do not invent details.
Preserve every explicitly requested exception or extra edit. Do not add edits, framing changes, reference pose, or reference body proportions. Do not describe background or lighting unless the user requests changing them.
The source video supplies pose, motion and shoulder silhouette; this is an editing instruction, not a new scene description.
No explanation, reasoning, markdown, labels, or generic quality/preservation paragraph."""

MIMO_MODEL = "MiMo-VL-7B-RL-2508"


def pe_available(backend):
    return backend == "mimo" or bool(os.environ.get("OPENAI_API_KEY"))


def pe_model_name(backend, model=None):
    return MIMO_MODEL if backend == "mimo" else model or os.environ.get("PE_MODEL") or DEFAULT_MODEL


def _person_replacement(prompt):
    """Use the short template only for an explicit whole-person replacement."""
    text = prompt.lower()
    if re.search(r"(?:不要|别|勿|do not|don't|never)[^，。,.!;\n]*(?:换人|替换|换成|replace|swap)", text):
        return False
    scopes = re.findall(r"\b(?:replace|swap)\s+(.+?)\s+(?:with|by|for)\s+", text)
    scopes += re.findall(r"([^，。！？;\n]+?)(?:替换成|替换为|换成|变成)", text)
    for scope in scopes:
        if re.search(r"衣|服|裤|裙|帽|眼镜|头发|发型|脸|背景|颜色|\b(?:clothes|clothing|outfit|shirt|top|dress|jacket|pants|trousers|hat|hair|face|background|color)\b", scope):
            continue
        if re.search(r"人物|主角|男人|女人|男孩|女孩|人|我|\b(?:person|people|man|woman|girl|boy|subject|character|actor|individual|me)\b", scope):
            return True
    return text.strip() in {"换人", "替换人物", "person replacement", "swap person"}


def _mimo_prompt(task_type, user_prompt):
    if task_type == "rv2v":
        if _person_replacement(user_prompt):
            return MIMO_PERSON_SYSTEM_PROMPT, "User request: " + user_prompt, "person_short", 128
        return RV2V_SYSTEM_PROMPT, RV2V_TEMPLATE.format(user_prompt=user_prompt), "rv2v", 384
    return SYSTEM_PROMPT, V2V_TEMPLATE.format(user_prompt=user_prompt), "v2v", 384


class MimoPromptEnhancer:
    """Use the runtime's existing generator; never load a second set of weights."""

    model = MIMO_MODEL

    def __init__(self, pipeline, lock):
        if pipeline is None or lock is None:
            raise ValueError("Local PE requires a loaded pipeline and its inference lock")
        self.pipeline = pipeline
        self.lock = lock
        self.template = None

    def _generate(self, task_type, user_prompt, video, ref_image, stop, deadline):
        import time
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set() or time.monotonic() >= deadline

        def image(item):
            if isinstance(item, tuple):
                item = item[0]
            if isinstance(item, str):
                with Image.open(item) as im:
                    return _downscale(im.convert("RGB"))
            if not isinstance(item, Image.Image):
                raise TypeError("Local PE expects PIL images or local image paths")
            return _downscale(item.convert("RGB"))

        frames = video if isinstance(video, list) else ([video] if video is not None else [])
        images = [image(item) for item in frames]
        if task_type == "rv2v":
            if ref_image is None:
                raise ValueError("Reference editing PE requires a reference image")
            images.insert(0, image(ref_image))
        system, text, self.template, max_tokens = _mimo_prompt(task_type, user_prompt)
        content = [{"type": "text", "text": text}]
        for i in range(len(images)):
            content.extend([{"type": "text", "text": f"\n[Image {i}]:"}, {"type": "image"}])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        if not self.lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise TimeoutError("Local PE timed out waiting for the encoder")
        try:
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("Local PE cancelled before generation")
            model, processor = self.pipeline.text_encoder, self.pipeline.qwen_processor
            device = self.pipeline._execution_device
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rendered += "<think>\n\n</think>\n\n"
            inputs = processor(text=[rendered], images=images or None, padding=True,
                               return_tensors="pt").to(device)
            rope_owner = getattr(model, "model", model)
            has_rope = hasattr(rope_owner, "rope_deltas")
            old_rope = getattr(rope_owner, "rope_deltas", None)
            try:
                from xvideo.serving.pe_aoti import generate_mimo_aoti
                with torch.inference_mode():
                    tokens = generate_mimo_aoti(
                        model, inputs, max_tokens,
                        lambda: stop.is_set() or time.monotonic() >= deadline,
                    )
                    self.decoder = "aoti" if tokens is not None else "eager"
                    if tokens is None:
                        output = model.generate(
                            **inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True,
                            disable_compile=True,
                            stopping_criteria=StoppingCriteriaList([Stop()]),
                        )
                        tokens = output[0, inputs.input_ids.shape[1]:].cpu()
            finally:
                if has_rope:
                    rope_owner.rope_deltas = old_rope
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("Local PE exceeded its generation deadline")
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else (eos or [])
            if tokens.numel() >= max_tokens and int(tokens[-1]) not in eos:
                raise ValueError("Local PE output was truncated")
            text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if not text or "<think" in text.lower() or "</think" in text.lower():
                raise ValueError("Local PE returned reasoning instead of an editing prompt")
            return _sanitize_enhanced(text, user_prompt)
        finally:
            self.lock.release()

    async def enhance(self, task_type, user_prompt, video=None, ref_image=None, *, timeout=90.0):
        import threading
        import time

        if not user_prompt or not user_prompt.strip():
            return user_prompt
        stop = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(
            self._generate, task_type, user_prompt, video, ref_image, stop, time.monotonic() + timeout,
        ))
        try:
            return await asyncio.wait_for(asyncio.shield(worker), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            stop.set()
            # CUDA work cannot be killed mid-kernel. Drain before handing this
            # same encoder back to video generation or another session.
            try:
                await asyncio.shield(worker)
            except Exception:
                pass
            raise
