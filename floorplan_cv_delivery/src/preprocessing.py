"""Input decoding, orientation handling, PDF rendering, and tile bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps


@dataclass(frozen=True)
class PageImage:
    image: Image.Image
    page_index: int | None
    source_width: int
    source_height: int
    source_kind: str
    pdf_page_width_points: float | None = None
    pdf_page_height_points: float | None = None


@dataclass(frozen=True)
class Tile:
    image: Image.Image
    x: int
    y: int
    width: int
    height: int


def load_pages(source: str | Path | bytes | bytearray | BinaryIO | Image.Image, dpi: int = 200) -> list[PageImage]:
    """Decode PNG/JPEG or render every PDF page, without cropping page content."""
    if dpi < 36 or dpi > 1200:
        raise ValueError("PDF dpi must be between 36 and 1200")
    if isinstance(source, Image.Image):
        image = ImageOps.exif_transpose(source).convert("RGB")
        return [PageImage(image, None, image.width, image.height, "image")]

    name = getattr(source, "name", "")
    if isinstance(source, (str, Path)):
        path = Path(source)
        name = path.name
        payload: str | Path | BytesIO = path
    else:
        raw = source if isinstance(source, (bytes, bytearray)) else source.read()
        payload = BytesIO(raw)

    is_pdf = str(name).lower().endswith(".pdf")
    if not is_pdf and isinstance(payload, BytesIO):
        is_pdf = payload.getvalue().startswith(b"%PDF-")
    if is_pdf:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF input requires PyMuPDF (pymupdf)") from exc
        if isinstance(payload, (str, Path)):
            doc = fitz.open(payload)
        else:
            doc = fitz.open(stream=payload.getvalue(), filetype="pdf")
        pages: list[PageImage] = []
        scale = dpi / 72.0
        for index, page in enumerate(doc):
            rect = page.rect
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, annots=True)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(PageImage(image, index, image.width, image.height, "pdf", rect.width, rect.height))
        if not pages:
            raise ValueError("PDF contains no pages")
        return pages

    try:
        with Image.open(payload) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise ValueError("Input must be a readable PNG, JPG, JPEG, or PDF") from exc
    return [PageImage(image, None, image.width, image.height, "image")]


def make_tiles(image: Image.Image, tile_size: int = 1024, overlap: int = 128) -> list[Tile]:
    """Cover the complete image with overlapping tiles, including right/bottom edges."""
    if tile_size < 256:
        raise ValueError("tile_size must be at least 256 pixels")
    if overlap < 0 or overlap >= tile_size // 2:
        raise ValueError("overlap must be non-negative and less than half tile_size")
    if image.width <= tile_size and image.height <= tile_size:
        return [Tile(image, 0, 0, image.width, image.height)]
    stride = tile_size - overlap
    xs = list(range(0, max(1, image.width - tile_size + 1), stride))
    ys = list(range(0, max(1, image.height - tile_size + 1), stride))
    if xs[-1] + tile_size < image.width:
        xs.append(image.width - tile_size)
    if ys[-1] + tile_size < image.height:
        ys.append(image.height - tile_size)
    return [
        Tile(image.crop((x, y, min(x + tile_size, image.width), min(y + tile_size, image.height))), x, y,
             min(tile_size, image.width - x), min(tile_size, image.height - y))
        for y in ys for x in xs
    ]
