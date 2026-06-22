from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from apertura.config import get_settings


def render_pdf(pdf_path: str | Path) -> list[Image.Image]:
    """Render every page of a PDF to a PIL image.

    Requires the system `poppler` binary (see README).
    """
    settings = get_settings()
    kwargs = {"dpi": settings.pdf_dpi}
    if settings.poppler_path:
        kwargs["poppler_path"] = settings.poppler_path
    return convert_from_path(str(pdf_path), **kwargs)
