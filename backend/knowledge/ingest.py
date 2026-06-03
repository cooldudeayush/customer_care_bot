"""One-time corpus ingestion: embed the policy docs and persist the index.

Run ONCE after setting GEMINI_API_KEY (from the backend/ directory):

    python -m knowledge.ingest

This embeds data/corpus/*.md with Gemini embeddings and writes the vector index
to data/vectorstore/index.json. The server then loads that index at startup —
we embed the corpus once and reuse it (free-tier friendly). Re-run after editing
the corpus.
"""

from __future__ import annotations

import asyncio
import logging

from knowledge.retriever import retriever
from llm import LLMNotConfigured

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    try:
        n = await retriever.build()
    except LLMNotConfigured:
        print(
            "✗ GEMINI_API_KEY is not set. Add it to backend/.env, then re-run "
            "`python -m knowledge.ingest`."
        )
        return
    if n == 0:
        print("✗ No corpus documents found. Add markdown files under data/corpus/.")
        return
    retriever.save()
    print(f"✓ Ingested {n} chunks and saved the retrieval index.")


if __name__ == "__main__":
    asyncio.run(main())
