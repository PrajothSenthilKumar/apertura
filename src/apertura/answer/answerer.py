import base64

from anthropic import Anthropic

from apertura.config import get_settings

SYSTEM_PROMPT = (
    "You are a financial-document analyst. Answer the user's question using ONLY "
    "the information visible in the provided document page images. The answer is "
    "often inside a table or chart. Quote exact figures. If the answer is not "
    "present on these pages, say so plainly. Be concise."
)


def _encode_image(path: str) -> dict:
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


def answer_question(question: str, page_paths: list[str]) -> str:
    """Send the question + retrieved page images to Claude and return the answer."""
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)

    content = [_encode_image(p) for p in page_paths]
    content.append({"type": "text", "text": question})

    message = client.messages.create(
        model=settings.answer_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in message.content if block.type == "text")