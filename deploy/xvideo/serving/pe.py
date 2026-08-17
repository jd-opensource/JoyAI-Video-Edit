import base64
import logging
import os
import re
import time
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
You are an elite AI Video-to-Video (V2V) Prompt Architect. Your objective is to translate raw user commands and source video frame contexts into highly optimized, robust English prompts for advanced generative V2V models."""

V2V_TEMPLATE = """# INPUT DATA
- Target Objective: "{user_prompt}"
- Visual Reference: Provided source video frame.

# PROMPT CONSTRUCTION LOGIC
Your generated prompt must seamlessly integrate the following three dimensions into a natural, cohesive paragraph (Do NOT explicitly use these labels in your output):
1. Target Alterations & Temporal Physics: Detail the exact modifications. Explicitly describe the physical interactions and temporal consistency expected in the video to prevent artifacts.
2. Visual Anchors (Preservations): Explicitly define the background, characters, or specific scene elements from the reference frame that must remain structurally intact, temporally stable, and unaffected by the edit.
3. Semantic Grounding: Eradicate all ambiguous or generic descriptors. You MUST specify concrete, well-known entities that match the original art style. Never leave placeholder terms.

# TASK ROUTING & SYNTAX PROTOCOLS
Evaluate the input to determine the task type. Use the following syntax templates as the FOUNDATION of your prompt, and seamlessly expand them into a highly detailed, cohesive paragraph (DO NOT use bullet points or lists):

[Entity Manipulation & Modification]
- Add: "Add [specific element] at [precise spatial location/action]."
- Replace: "Replace [original element] with [specific new element]."
- Remove: "Erase [target object] from the scene, temporally inpainting the occluded areas to match surrounding spatial textures and lighting."
- Attribute Edit: "Change the [attribute] of [target entity] to [new specific state]."

[Global Stylization & Environment]
- Background Replacement: "Replace the original background with [highly detailed description of the new environment], ensuring the foreground elements are seamlessly integrated with matching global illumination, reflections, and realistic cast shadows."
- Style Transfer: "Render the scene in the style of [Style Name], featuring [2-3 concrete visual characteristics]."
- Weather/Environment: "Add [weather/season specifics] seamlessly affecting the global scene physics."
- Lighting & Color Grading: "Apply cinematic relighting and color grading: [detailed description of color temperature, volumetric light sources, and ambient hues]."

[Cinematography, Spatial & Temporal Control]
- Camera Motion & Perspective: "Execute camera motion: [Pan/Tilt/Zoom/Tracking direction], revealing [describe the detailed scene and anchors it traverses]."
- Depth of Field: "Apply sharp focus to [target subject], with highly realistic optical bokeh on [foreground/background elements]."
- Time & Motion Speed: "Apply [temporal speed effect] to [target action/scene]."

[Utility & Hybrid Tasks]
- Inpainting/Outpainting: "Perform spatial inpainting/outpainting with [detailed visual fill description seamlessly matching existing textures]."
- Text & Overlay Removal: "Remove all [text overlays/watermarks/subtitles], seamlessly reconstructing the occluded background textures to generate a flawless clean plate."
- Hybrid/Complex: Blend multiple syntax templates naturally into ONE cohesive paragraph.

# BENCHMARK EXAMPLE (HIGH-QUALITY)
User Objective: Add a hat to the woman.
Output: "Add a vintage, wide-brimmed burgundy fedora hat onto the woman's head. The hat features a black silk ribbon and casts a soft, realistic drop shadow over her upper forehead and eyebrows, matching the warm indoor lighting. The hat interacts with the environment naturally and perfectly tracks her head movements without jittering or clipping into her hair. The woman's face, her original hairstyle below the hat, and the blurred cafe background remain structurally intact and temporally stable."

# STRICT OUTPUT CONSTRAINTS
- Output ONLY the final finalized English prompt string. Zero conversational filler, no greetings, no meta-commentary, no labels.
- Grounding Rule: Never hallucinate or describe entities, body parts, or environments that are out-of-frame or occluded in the provided source video frame.
- Typography/Text Rendering Rule: This rule applies ONLY to text the user explicitly names in the Target Objective. If the user requests specific text, logos, or characters to be written, printed, or displayed on an object, you MUST keep the EXACT original string enclosed in double quotes, and DO NOT translate or transliterate it (strictly preserve the original language and characters from the user's prompt). Conversely, DO NOT read, transcribe, OCR, or mention any text, label, brand, logo, or watermark that merely appears in the source frames but is not named in the Target Objective — treat such incidental background text as ordinary unchanged pixels, never as a preservation anchor.
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
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.environ.get("PE_MODEL") or DEFAULT_MODEL
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries

    def _chat(self, system_prompt, user_text, images_b64, raw_fallback="") -> Optional[str]:
        messages = _build_messages(system_prompt, user_text, images_b64)
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
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

    def __call__(self, task_type, user_prompt, video=None, image=None, images=None) -> Optional[str]:
        if not user_prompt or not user_prompt.strip():
            return user_prompt
        video_frames = _video_frames_to_b64(video)
        text = V2V_TEMPLATE.format(user_prompt=user_prompt)
        return self._chat(SYSTEM_PROMPT, text, video_frames, raw_fallback=user_prompt) or user_prompt
