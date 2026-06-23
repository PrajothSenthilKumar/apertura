from __future__ import annotations
from enum import Enum
from anthropic import Anthropic
from apertura.config import get_settings

ROUTER_PROMPT = """You are a document-query classifier.
Classify the following question as either:
- "visual" if the answer is most likely found in a TABLE, CHART, FIGURE, GRAPH, or FINANCIAL STATEMENT.
- "text" if the answer is most likely found in plain PROSE or NARRATIVE text.
Reply with exactly one word: visual  OR  text
Question: {question}"""

class QueryType(str, Enum):
    VISUAL = "visual"
    TEXT = "text"

def classify_query(question: str) -> QueryType:
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=5,
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(question=question)}],
    )
    label = "".join(b.text for b in resp.content if b.type == "text").strip().lower()
    return QueryType.VISUAL if "visual" in label else QueryType.TEXT