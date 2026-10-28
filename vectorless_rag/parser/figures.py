"""What every parser does with the figures it finds: drop the small ones, host the rest.

PDF, docx and xlsx all end the same way: an ``image`` item whose ``img_path`` is a
local file is uploaded to S3 and now points at a URL the VLM can fetch. A ready-made
content list (``.json``) gets the same guarantee through :func:`fit_figures`. The size
thresholds are one setting for all of them (``FIGURE_MIN_*`` in ``config.py``).
"""
import io
import logging
from pathlib import Path
import httpx
from PIL import Image
from ..config import settings
from ..storage.s3 import upload_figures_to_s3

logger = logging.getLogger(__name__)

# Long side of every image sent to the VLM: small text inside figures and slides stays readable.
FIGURE_MAX_PIXELS = 3840
# The VLM gateway (Bedrock) rejects a request body from ~2.5 MB of image file
# ("Failed to buffer the request body"); 2 MB leaves room for base64 and the prompt.
FIGURE_MAX_BYTES = 2_000_000
# Quality first, then size. 4:4:4 (subsampling=0) keeps thin coloured lines and small
# coloured text sharp; the default 4:2:0 quarters the colour resolution.
# ponytail: PNG would be lossless for screenshots, but slides.py sends these as image/jpeg.
JPEG_QUALITIES = (95, 90, 80, 70)


def is_large_enough(width_px: int, height_px: int, area_ratio: float | None = None) -> bool:
    """Whether a figure clears the size thresholds."""
    return (
        width_px >= settings.figure_min_width_px
        and height_px >= settings.figure_min_height_px
        and (area_ratio is None or area_ratio >= settings.figure_min_area_ratio)
    )


def is_large_image(data: bytes, name: str) -> bool:
    """:func:`is_large_enough` for embedded image bytes (docx, xlsx)."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception as e:
        logger.warning(f"Gambar {name} dilewati: tidak bisa dibaca ({type(e).__name__})")
        return False
    return is_large_enough(width, height)


def cap_image_size(img: Image.Image) -> Image.Image:
    """Resize ``img`` so its longest side is at most ``FIGURE_MAX_PIXELS``.

    Uses ``thumbnail`` (in-place, aspect-preserving). Returns the same object.
    No-op when the image already fits.
    """
    if max(img.width, img.height) > FIGURE_MAX_PIXELS:
        img.thumbnail((FIGURE_MAX_PIXELS, FIGURE_MAX_PIXELS), Image.Resampling.LANCZOS)
    return img


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def fits_vlm(img: Image.Image, size_bytes: int) -> bool:
    """Whether an image file can go to the VLM as it is."""
    return max(img.size) <= FIGURE_MAX_PIXELS and size_bytes <= FIGURE_MAX_BYTES


def _flatten(img: Image.Image) -> Image.Image:
    """``img`` in a mode JPEG can store; transparent areas become white, not black."""
    if img.mode in ("RGB", "L"):
        return img
    rgba = img.convert("RGBA")
    background = Image.new("RGB", rgba.size, "white")
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def save_for_vlm(img: Image.Image, path: str | Path, original: bytes | None = None) -> Path:
    """Write ``img`` so the VLM can take it: long side ≤ ``FIGURE_MAX_PIXELS``, file ≤ ``FIGURE_MAX_BYTES``.

    ``original`` — the embedded file of a docx/xlsx image — is written byte for byte
    when it already fits both limits. Otherwise the image is re-encoded as JPEG,
    lowering the quality first and then the size, until it fits. Returns the path
    written, which ends in ``.jpg`` when the image was re-encoded.
    """
    path = Path(path)
    if original is not None and fits_vlm(img, len(original)):
        path.write_bytes(original)
        return path
    img = cap_image_size(_flatten(img))
    while True:
        for quality in JPEG_QUALITIES:
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=quality, subsampling=0)
            if buffer.tell() <= FIGURE_MAX_BYTES:
                path = path.with_suffix(".jpg")
                path.write_bytes(buffer.getvalue())
                return path
        img = img.resize((max(1, img.width * 3 // 4), max(1, img.height * 3 // 4)), Image.Resampling.LANCZOS)


def fit_figures(content_list: list[dict], tmp_dir: Path) -> None:
    """Bring the images of a ready-made content list (``.json``) under the VLM limits, in place.

    Parsers write their figures through :func:`save_for_vlm`; a ready-made list points at
    images made elsewhere. Each is read once. One that fits keeps its ``img_path``, so a
    durable S3 URL stays as it is; one that does not is re-encoded into ``tmp_dir`` for
    :func:`host_figures` to upload.

    Raises:
        RuntimeError: an image cannot be downloaded or read. Raised rather than skipped,
            so the document is never indexed without it.
    """
    for index, item in enumerate(content_list):
        source = str(item.get("img_path") or "")
        if item.get("type") != "image" or not source:
            continue
        try:
            if _is_url(source):
                response = httpx.get(source, timeout=60, follow_redirects=True)
                response.raise_for_status()
                data = response.content
            else:
                data = Path(source).read_bytes()
            with Image.open(io.BytesIO(data)) as img:
                if not fits_vlm(img, len(data)):
                    item["img_path"] = str(save_for_vlm(img, tmp_dir / f"figure_{index}.jpg"))
        except Exception as e:
            raise RuntimeError(f"Gambar tidak bisa dibaca: {source}") from e


def host_figures(content_list: list[dict]) -> None:
    """Upload every local ``img_path`` of ``content_list`` to S3, in place; URLs are already hosted.

    That is every ``image`` item, plus the ``text`` items of a slide deck (``slides.py``).
    """
    upload_figures_to_s3([item for item in content_list if item.get("img_path") and not _is_url(item["img_path"])])
