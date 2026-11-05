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
change and exception after applying mandatory edit-type rules; completeness takes priority over
the word target. Put the main edit first, add only useful visual details, and end with one brief
preservation clause when needed. No explanations, labels, lists, reasoning, generic quality
claims or repeated negative commands.

# SCOPE AND GROUNDING
The user's explicit edits, exclusions and requested strength override defaults, but not mandatory
edit-type rules. Identify the combined edit scope before deciding what to preserve. Keep
attributes outside that scope; never preserve an attribute that another requested edit changes.
Use the source frames to identify visible targets, clothing layers, scene layout and framing.
Do not invent source objects, hidden anatomy or occluded details. Newly requested objects,
appearances and environments may differ from the source; describe only details needed to define
the requested result. A broad request is not permission to add unrelated objects or edits.
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

- Clothing and footwear: change only the requested clothing item, layer or footwear. Keep
  clothing, footwear and accessories outside the replacement scope, along with source identity
  and body proportions. Fit new garments naturally to create a flattering, slender silhouette
  while preserving the requested loose or tailored cut. A color-only edit retains cut or shape.
  Trouser-only edits retain waistband height and the top's hem, drape and overlap unless
  requested otherwise. Describe requested sleeve length or trouser flare as a garment
  construction and silhouette change, not an extra layer; match the requested cut and degree of
  flare, include "full sleeves reach the hand bases" for long-sleeved garments. Lantern sleeves
  have rounded volume tapering to narrow wrist openings at the hand bases. For qipao/cheongsam
  only, retain requested collar, fastenings and fabric; explicitly describe a continuous opaque
  skirt fully closed around both legs, all panels joined to the hem throughout motion regardless
  of requested slits; omit offscreen details. For requested apricot/杏色 clothing/footwear,
  substitute "warm off-white" for every occurrence of the requested color name; never write
  "apricot". An explicit orange/fruit hue, color code or precise shade takes priority; use the
  specified color without calling it "warm off-white".

- Background replacement: directly instruct replacing the background with the requested scene,
  adding only details needed to define the new environment. Replace the original scene content
  within that scope, honoring explicit exceptions. Briefly preserve unaffected subjects, motion
  and framing without describing incidental source appearance or fixing a particular pose.
  Keep an already clear request concise; omit source-object inventories and compositing steps.
  Restyling an existing background changes its appearance, not its objects or layout, unless requested.

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

RV2V_SYSTEM_PROMPT = """Write precise English instructions for reference-image-guided video editing.
The source supplies motion, framing and scene; the reference supplies only requested attributes.
Explicit user instructions override defaults, but not mandatory edit-type rules."""

RV2V_TEMPLATE = """# INPUT
User request: {user_prompt}
Image 0 is the REFERENCE IMAGE; subsequent images are SOURCE VIDEO FRAMES.

# OUTPUT AND SCOPE
Return only one cohesive English editing prompt, concise for simple requests. Start with the
requested edit; say "the reference image", never image numbers. Describe only useful, clearly
visible reference features. Preserve every requested edit, exclusion, strength and timing;
these override defaults, but not mandatory edit-type rules. Attribute-only edits change only
that attribute. Do not import unrelated reference identity, physique, pose, framing, lighting,
background or props. Express exclusions as properties of the requested result; give removal instructions
only for visible source objects. Do not invent hidden anatomy, occluded details or objects to remove.

# EDIT TYPES
- Clothing and footwear: change only the requested clothing item, layer or footwear. For an
  unqualified top replacement, also replace layers concealing it, including outerwear and
  shoulder-draped coverings, unless explicitly retained. Keep clothing, footwear and accessories
  outside the replacement scope, along with source identity and body proportions. Replace
  targeted items with the reference garment, explicitly naming "the reference image", matching
  cut or shape, color, material and pattern without blending in the source item's details. Fit
  new garments naturally to create a flattering, slender silhouette while preserving the
  reference garment's loose or tailored cut unless otherwise requested. Trouser-only edits
  retain waistband height and the top's hem, drape and overlap. Match sleeve length and degree
  of trouser flare through garment construction, not extra layers; include "full sleeves reach
  the hand bases" for long-sleeved garments. Lantern sleeves have rounded volume tapering to
  narrow wrist openings at the hand bases. For qipao/cheongsam only, retain reference collar,
  fastenings and fabric; explicitly describe a continuous opaque skirt fully closed around both
  legs, all panels joined to the hem throughout motion regardless of requested/reference slits.
  Use user-specified clothing/footwear colors; otherwise use reference colors. For requested
  apricot/杏色 colors, write only "warm off-white", never "apricot"; explicit orange/fruit hues,
  precise shades or color codes override. Other items retain their colors.

- Face-only replacement (explicit face/head requests): transfer reference identity and hair/head
  features except explicitly retained attributes. Keep source clothing and body; do not import
  reference outfits or props. Hair-only edits do not change facial identity.

- Whole-person/character/creature replacement: requests such as "replace me with this man/woman"
  transfer reference identity, hair and visible outfit together by default, not just the face.
  Describe both head/face and outfit features. Replace source clothing with the reference outfit;
  do not retain or blend its design unless explicitly requested. Preserve source pose, motion
  and body proportions unless edited. A costume change alone does not replace identity.

- Background replacement: directly instruct replacing the background with the environment in
  the reference image, adding only visible details needed to define it. Replace the original
  scene content within that scope, honoring explicit exceptions. Briefly preserve unaffected
  subjects, motion and framing without describing incidental source appearance or fixing a
  particular pose. Keep an already clear request concise; omit source-object inventories and
  compositing steps. Do not add reference people unless requested. Restyling an existing
  background changes its appearance, not its objects or layout, unless requested.

- Objects/accessories: transfer the requested item at its functional position and appropriate scale.

- Style: apply reference style within the requested scope and strength. Preserve recognizable
  identity, action and layout without restoring the original rendering in styled regions.

# TEXT
Render user-specified text verbatim in double quotes, preserving language and spelling even if
originally unquoted. Do not translate it or copy incidental source/reference text, logos or
watermarks unless requested. For removal, identify the text's scope without repeating its wording.

# VIDEO AND PRESERVATION
Apply the edit from the first frame unless gradual or timed changes are requested. Follow each
source frame's pose, shoulder contour, arm positions, scale and motion; never freeze the anchor
pose or copy the reference stance. Maintain tracking, occlusion, garment deformation and natural
layering/contact around retained accessories, with appropriate shadows and reflections.
Respect source framing; never zoom out to show an edit. A complete outfit applies as its parts
become visible later. Retain explicitly requested details, but elaborate only currently visible,
image-supported features; do not invent offscreen anatomy or garment details.
End with a brief preservation clause for unaffected attributes, including the background when
unedited. Never preserve or restore source appearances replaced, covered or transformed by any
requested edit.
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


QWEN_MODEL = "Qwen3.5-35B-A3B-GPTQ-Int4"


def pe_available(backend):
    return backend == "qwen" or bool(os.environ.get("OPENAI_API_KEY"))


def pe_model_name(backend, model=None):
    return QWEN_MODEL if backend == "qwen" else model or os.environ.get("PE_MODEL") or DEFAULT_MODEL


class QwenPromptEnhancer:
    """Use the Qwen model prepared once with the video runtime."""

    model = QWEN_MODEL

    def __init__(self, pipeline, lock):
        if pipeline is None or lock is None:
            raise ValueError("Local PE requires a loaded pipeline and its inference lock")
        self.pipeline = pipeline
        self.lock = lock
        self.template = None

    def _generate(self, task_type, user_prompt, video, ref_image, stop, deadline):
        import time

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
        self.template = task_type
        if task_type == "rv2v":
            system, text = RV2V_SYSTEM_PROMPT, RV2V_TEMPLATE.format(user_prompt=user_prompt)
        else:
            system, text = SYSTEM_PROMPT, V2V_TEMPLATE.format(user_prompt=user_prompt)
        if not self.lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise TimeoutError("Local PE timed out waiting for inference")
        try:
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("Local PE cancelled before generation")
            runtime = getattr(self.pipeline, "_pe_qwen", None)
            if runtime is None:
                raise RuntimeError("Qwen PE was not prepared at startup")
            text, timings = runtime.generate(
                system, text, images, lambda: stop.is_set() or time.monotonic() >= deadline,
            )
            self.decoder = "cuda_graph"
            self.timings = timings
            logger.info("Qwen PE %s: prefill %.3fs, graph capture %.3fs",
                        task_type, timings["prefill_seconds"], timings["capture_seconds"])
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("Local PE exceeded its generation deadline")
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
            # inference lock back to video generation or another session.
            try:
                await asyncio.shield(worker)
            except Exception:
                pass
            raise
