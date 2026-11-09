from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

router = APIRouter()

REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "rv2v_reference"
_LEGACY_REFERENCES = {
    "hat": "4e481f7a-2443-4935-a841-af6113cc4236.png",
    "scarf": "5e178546-3ebf-40df-bb86-01613dd96c3b.png",
    "pink_tee": "1c182f2f-32cf-4825-904e-64c69aed2e31.png",
    "orange_glasses": "486b9561-e73d-45ca-bb2d-2a47998a0a73.png",
}

def _reference_response(image_id: str, request: Request, thumbnail: bool = False):
    if image_id in _LEGACY_REFERENCES and not thumbnail:
        path = REFERENCE_ROOT / _LEGACY_REFERENCES[image_id]
    else:
        try:
            if str(uuid.UUID(image_id)) != image_id:
                raise ValueError
        except ValueError:
            raise HTTPException(status_code=404, detail="Reference image not found")
        directory = REFERENCE_ROOT / ("thumbnails" if thumbnail else "ip")
        path = directory / f"{image_id}{'.jpg' if thumbnail else '.png'}"
        if path.resolve().parent != directory.resolve():
            raise HTTPException(status_code=404, detail="Reference image not found")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Reference image not found")
    stat = path.stat()
    etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
    headers = {"Cache-Control": "public, max-age=86400", "ETag": etag}
    if request.headers.get("if-none-match") in (etag, "*"):
        return Response(status_code=304, headers=headers)
    return FileResponse(path, media_type="image/jpeg" if thumbnail else "image/png", headers=headers)

@router.get("/reference-images/{image_id}/thumbnail")
def reference_thumbnail(image_id: str, request: Request):
    return _reference_response(image_id, request, thumbnail=True)

@router.get("/reference-images/{image_id}")
def reference_image(image_id: str, request: Request):
    return _reference_response(image_id, request)
