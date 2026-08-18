from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Sequence


_IMAGE_MIME_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def user_input(text: str, image_paths: Sequence[str | Path] = ()) -> str | list[dict[str, Any]]:
    if not image_paths:
        return text
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    for path_value in image_paths:
        path = Path(path_value).expanduser().resolve()
        mime_type = _IMAGE_MIME_TYPES.get(path.suffix.lower())
        if mime_type is None:
            raise ValueError(f"Unsupported image type: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
                "detail": "auto",
            }
        )
    return [{"role": "user", "content": content}]


__all__ = ["user_input"]
