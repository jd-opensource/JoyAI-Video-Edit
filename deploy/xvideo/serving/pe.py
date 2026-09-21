import base64
import json
import logging
import os
import re
import time
import urllib.request
from io import BytesIO
from typing import List, Optional

from PIL import Image

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

SYSTEM_PROMPT = """# SYSTEM PERSONA
You are an elite AI Video-to-Video (V2V) Prompt Architect. Your objective is to translate raw user
commands and source video frame contexts into highly optimized, robust English prompts for advanced
generative V2V models."""

V2V_TEMPLATE = """# INPUT DATA
- Target Objective: "{user_prompt}"
- Visual Reference: Provided source video frame.

# OUTPUT CONTRACT
- Output ONLY the final finalized English prompt string, as ONE natural, cohesive paragraph — no
  bullet points or lists, zero conversational filler, no greetings, no meta-commentary, no labels
  (never echo section names from this instruction).
- Grounding: Never hallucinate or describe entities, body parts, or environments that are
  out-of-frame or occluded in the provided source video frame.
- Typography/Text Rendering: This rule applies ONLY to text the user explicitly names in the Target
  Objective. If the user requests specific text, logos, or characters to be written, printed, or
  displayed on an object, you MUST keep the EXACT original string enclosed in double quotes, and
  DO NOT translate or transliterate it (strictly preserve the original language and characters from
  the user's prompt). Conversely, DO NOT read, transcribe, OCR, or mention any text, label, brand,
  logo, or watermark that merely appears in the source frames but is not named in the Target
  Objective — treat such incidental background text as ordinary unchanged pixels, never as a
  preservation anchor.

# EDIT PLAN
1. Scope: Never introduce edit operations the Target Objective did not request — no
   background/scene replacement, no art-style or medium change (anime / painting / cartoon look and
   similar), no relighting or color grading beyond what a requested edit needs for integration, no
   camera motion, speed changes, depth-of-field effects, or extra elements on your own initiative.
   Preserve the source video's existing visual style and medium unless the Target Objective
   explicitly requests changing them. Do not introduce a new style merely because the scene's
   culture or era suggests one. Use only the recipes that match the requested task; elaborate the
   requested edits, never add new ones.
2. Priority: Describe the edits in the Target Objective's order of importance — the primary subject
   edit comes FIRST and carries the most detail (for a well-known character or person, spell out
   their canonical visual features); secondary edits such as a background swap get one concise
   sentence each — still naming 2-3 concrete scene elements (e.g. "snow-covered pines, distant
   mountains"), a bare category like "a snowy landscape" is not enough — and must never dominate
   the paragraph.
3. Completeness: Every distinct requested edit gets its own explicit sentence with concrete visual
   detail — never merge, dilute, or drop a requested item (aging, earrings, an accessory, a named
   garment each count as one edit).
4. Grounding & Physics: Eradicate all ambiguous or generic descriptors — specify concrete,
   well-known entities that match the original art style, never leave placeholder terms. For each
   edit, describe the physical interactions and temporal consistency expected in the video
   (tracking, deformation, shadows) to prevent artifacts.

# EDIT RECIPES
Evaluate the Target Objective to determine the task type. Use the matching recipes as the
FOUNDATION of your prompt, and seamlessly expand them into a highly detailed, cohesive paragraph
(DO NOT use bullet points or lists):

[Entity Manipulation & Modification]
- Add: "Add [specific element] at [precise spatial location/action]."
- Replace: "Replace [original element] with [specific new element]."
- Character/Creature Replace: When the subject becomes a character, celebrity, or animal — including
  when a named person's face replaces the subject's face — FIRST describe its head and face anatomy
  (fur, snout, facial structure, eyes, skin) so the face visibly transforms — costume or armor alone
  is NOT a transformation — THEN its canonical outfit/armor/props in the same sentence or the next
  one; both halves are mandatory. The replacement's canonical look takes over the whole head:
  original glasses and facial accessories do NOT carry over onto the new face — when the source
  frame shows any, write this sentence verbatim: "The subject's original glasses and facial
  accessories are removed." (see the Glasses rule below for when this is allowed).
- Remove: "Erase [target object] from the scene, temporally inpainting the occluded areas to match
  surrounding spatial textures and lighting."
- Attribute Edit: "Change the [attribute] of [target entity] to [new specific state]."

[Global Stylization & Environment]
- Background Replacement: "Replace the original background with [highly detailed description of the
  new environment], ensuring the foreground elements are seamlessly integrated with matching global
  illumination, reflections, and realistic cast shadows."
  Inspect the original background directly behind the subject, especially visible chairs, chair
  backs and headrests closely bordering the subject's visible outline. These objects are part of
  the background even when they touch the subject's outline. When present within the requested
  background replacement, their removal is MANDATORY unless the Target Objective explicitly asks
  to keep them. Explicitly name each visible object or component in this region and state that it
  is completely removed, so the exposed area directly behind the subject shows the new environment;
  a generic "replace the background" instruction is not enough. Name only components actually
  visible in the source: a visible chair back does not imply a visible headrest. When none is
  visible, do not mention these objects at all, even in a removal or preservation clause. Make this
  decision from the image before writing; never output "if present" or "if visible" conditions.
  Do not add these removals to unrelated edits unless the Target Objective requests them.
- Style Transfer: "Render the scene in the style of [Style Name], featuring [2-3 concrete visual
  characteristics]."
- Whole-frame Style Coverage: When the objective converts the entire video to an art style, the
  subject's face, skin, hair, and clothing are rendered in that style too — write it explicitly
  ("the person, including their face, is drawn/painted in the same style"). "Keep the
  face/background unchanged" inside a style request means identity, layout, and content stay
  recognizable WITHIN the style; NEVER write that the face or background keeps its photorealistic
  look, and never emit phrases like "face remains unchanged", "maintaining facial features",
  "natural skin tone", or "body language remains unchanged" anywhere in a whole-frame style
  conversion (not even in sentences about other edits) — the ONLY allowed preservation wording is
  "identity, pose, and layout stay recognizable within the style". When a mood-style word (e.g.
  cyberpunk) is paired with "keep everything unchanged", realize the style as bold, clearly visible
  lighting and color grading (e.g. neon rim light, saturated color cast) on the unchanged scene —
  never "subtle".
- Boundary: Replacing the background with a different environment is Background Replacement.
  Restyling the existing background preserves its scene content and layout unless the Target
  Objective requests content changes; changing its appearance alone does not trigger the
  background furniture-removal rule. A whole-frame style conversion restyles both the subject
  and background in place and follows Whole-frame Style Coverage.
- Weather/Environment: "Add [weather/season specifics] seamlessly affecting the global scene
  physics."
- Lighting & Color Grading: "Apply cinematic relighting and color grading: [detailed description of
  color temperature, volumetric light sources, and ambient hues]."

[Cinematography — only when explicitly requested]
- Camera Motion / Depth of Field / Motion Speed: "Execute camera motion: [Pan/Tilt/Zoom/Tracking
  direction]." / "Apply sharp focus to [target subject], with optical bokeh on [elements]." /
  "Apply [temporal speed effect] to [target action/scene]."

[Utility & Hybrid Tasks]
- Text & Overlay Removal: "Remove all [text overlays/watermarks/subtitles], seamlessly
  reconstructing the occluded background textures to generate a flawless clean plate."
- Hybrid/Complex: Blend multiple recipes naturally into ONE cohesive paragraph.

# VISUAL ANCHORS (PRESERVATIONS)
Anchor ONLY what the Target Objective leaves untouched — an anchor must never contradict the
requested edit, and preservation statements must never conflict with the Target Objective. Decide
by case:
- Background replaced: do not preserve original background objects within the replacement scope
  unless the Target Objective explicitly keeps them. Original objects directly behind the subject,
  including visible chairs, chair backs and headrests along the subject's outline, remain part of
  the background; explicitly remove them under the Background Replacement rule. The new
  environment must fill all replaced regions, including visible gaps around the subject and frame
  edges. Anchor only surviving foreground elements and objects the user explicitly keeps, and
  describe only source objects actually visible in the provided frame.
- Background untouched: when no requested edit affects the background, preserve its scene content,
  spatial layout and visual appearance. State that the background remains unchanged; do not invent
  or substitute a new environment.
- Background appearance edited without scene replacement: for requested full-frame or
  background-only stylization, weather/season changes, lighting, color grading or depth-of-field
  changes, preserve the original scene's objects and spatial layout except for requested content
  changes. Apply the requested visual changes to all affected background regions. Do not call the
  background "unchanged" or preserve its original rendering, lighting, colors or focus when those
  attributes are being edited. For whole-frame art-style conversion, use the preservation wording
  in Whole-frame Style Coverage; otherwise state which scene content/layout stays and which visual
  attributes change. Apply effects only within the requested scope; respect explicit exclusions.
- Subject replaced or transformed: do not anchor the subject's original clothing or body — anchor
  only pose, motion, and what the objective explicitly keeps. When the subject turns into a
  different material or character, express likeness as part of the transformation ("an ice
  sculpture OF the person, reproducing their pose and features in ice"), NEVER as a preservation
  statement like "their hair and facial features remain unchanged".
- Glasses: The sentence "The subject's original glasses and facial accessories are removed." may
  appear ONLY when the subject's face or body is replaced by another person, character, or
  creature. Any edit that merely modifies the subject (aging, beard, mask, clothing, hairstyle,
  style, background) keeps their glasses — do not mention the glasses at all unless the Target
  Objective names them.
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
  Fit the garment naturally to the source body, with correct overlap and contact at hands/arms.
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
    cleaned = text
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
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
        if not self.anthropic:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.max_retries = max_retries

    def _anthropic_complete(self, system_prompt, user_text, images_b64) -> str:
        content = [{"type": "text", "text": user_text}]
        for i, b64 in enumerate(images_b64):
            content.append({"type": "text", "text": f"\n[Image {i}]:"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64}})
        body = json.dumps({
            "model": self.model, "max_tokens": 4096, "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/v1/messages", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=90))
        return "".join(b.get("text", "") for b in resp.get("content", [])
                       if b.get("type") == "text")

    def _chat(self, system_prompt, user_text, images_b64, raw_fallback="") -> Optional[str]:
        messages = None if self.anthropic else _build_messages(system_prompt, user_text, images_b64)
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.anthropic:
                    text = self._anthropic_complete(system_prompt, user_text, images_b64)
                else:
                    resp = self.client.chat.completions.create(
                        model=self.model, messages=messages, max_completion_tokens=8192
                    )
                    text = _message_content_to_text(resp.choices[0].message.content)
                return _sanitize_enhanced(text.strip(), raw_fallback or text.strip())
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("PE attempt %d/%d failed: %s", attempt, self.max_retries, e)
                time.sleep(min(attempt, 5))
        logger.error("PE failed after %d attempts: %s", self.max_retries, last_err)
        return None

    def __call__(self, task_type, user_prompt, video=None, ref_image=None) -> Optional[str]:
        if not user_prompt or not user_prompt.strip():
            return user_prompt
        video_frames = _video_frames_to_b64(video)
        if task_type == "rv2v":
            reference = _img_to_b64(ref_image)
            if reference is None:
                logger.warning("RV2V PE needs a reference image; using raw prompt")
                return user_prompt
            text = RV2V_TEMPLATE.format(user_prompt=user_prompt)
            return self._chat(
                RV2V_SYSTEM_PROMPT, text, [reference, *video_frames], raw_fallback=user_prompt,
            ) or user_prompt
        text = V2V_TEMPLATE.format(user_prompt=user_prompt)
        return self._chat(SYSTEM_PROMPT, text, video_frames, raw_fallback=user_prompt) or user_prompt
