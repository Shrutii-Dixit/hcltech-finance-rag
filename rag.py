from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from ingest import COLLECTION_NAME, EMBEDDING_MODEL, ROOT_DIR, load_chroma_client, load_gemini_client


GENERATION_MODEL = "gemini-2.5-flash"
DEFAULT_TOP_K = 4
REFUSAL_TEXT = "I could not find this information in the provided reports."


@dataclass
class SourceItem:
    source: str
    page: int
    quarter: str


def load_collection() -> chromadb.Collection:
    client = load_chroma_client()
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Chroma collection '{COLLECTION_NAME}' is missing or empty. Run ingestion first."
        ) from exc


def normalize_quarter(value: str) -> str:
    match = re.search(r"(Q[1-4])[_\-\s]*FY(\d{2,4})", value, re.IGNORECASE)
    if not match:
        return value.strip()
    fy_value = match.group(2)
    if len(fy_value) == 4:
        fy_value = fy_value[-2:]
    return f"{match.group(1).upper()} FY{fy_value}"


def detect_requested_quarter(question: str) -> str | None:
    quarter_match = re.search(r"Q[1-4][\s\-_/]*FY\s?\d{2,4}", question, re.IGNORECASE)
    if quarter_match:
        return normalize_quarter(quarter_match.group(0))
    lowered = question.lower()
    if any(phrase in lowered for phrase in ("latest quarter", "most recent quarter", "current quarter")):
        return latest_indexed_quarter()
    return None


def latest_indexed_quarter() -> str | None:
    client = load_chroma_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        return None
    try:
        records = collection.get(include=["metadatas"], limit=100000)
    except Exception:  # noqa: BLE001
        return None

    quarter_values = {
        normalize_quarter(metadata.get("quarter", ""))
        for metadata in records.get("metadatas", [])
        if metadata and metadata.get("quarter")
    }
    if not quarter_values:
        return None
    return sorted(quarter_values, key=quarter_sort_key)[-1]


def quarter_sort_key(value: str) -> tuple[int, int]:
    match = re.search(r"Q([1-4])\s*FY(\d{2,4})", value, re.IGNORECASE)
    if not match:
        return (0, 0)
    quarter_number = int(match.group(1))
    fy_value = int(match.group(2))
    return (fy_value, quarter_number)


def load_api_client() -> genai.Client:
    load_dotenv(ROOT_DIR / ".env")
    return load_gemini_client()


def embed_question(client: genai.Client, question: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[question],
    )
    return response.embeddings[0].values


def flatten_query_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    rows: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        rows.append(
            {
                "id": chunk_id,
                "document": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "distance": distances[index] if index < len(distances) else None,
            }
        )
    return rows


def query_collection(
    collection: chromadb.Collection,
    question_embedding: list[float],
    top_k: int,
    quarter: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def add_results(query_result: dict[str, Any]) -> None:
        for row in flatten_query_result(query_result):
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            results.append(row)

    if quarter:
        try:
            add_results(
                collection.query(
                    query_embeddings=[question_embedding],
                    n_results=top_k,
                    where={"quarter": quarter},
                    include=["documents", "metadatas", "distances"],
                )
            )
        except Exception:  # noqa: BLE001
            pass

    if len(results) < top_k:
        try:
            add_results(
                collection.query(
                    query_embeddings=[question_embedding],
                    n_results=max(top_k, 6),
                    include=["documents", "metadatas", "distances"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Retrieval failed: {exc}") from exc

    if quarter:
        quarter_results = [row for row in results if normalize_quarter(row["metadata"].get("quarter", "")) == quarter]
        if quarter_results:
            return quarter_results[:top_k]

    return results[:top_k]


def build_context(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")
        quarter = metadata.get("quarter", "Unknown quarter")
        document = chunk.get("document", "")
        blocks.append(
            f"[{index}] Source: {source} | Quarter: {quarter} | Page: {page}\n{document}"
        )
    return "\n\n".join(blocks)


def build_sources(chunks: list[dict[str, Any]]) -> list[SourceItem]:
    sources: list[SourceItem] = []
    seen: set[tuple[str, int, str]] = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "Unknown source")
        page = int(metadata.get("page", 0) or 0)
        quarter = metadata.get("quarter", "Unknown quarter")
        key = (source, page, quarter)
        if key in seen:
            continue
        seen.add(key)
        sources.append(SourceItem(source=source, page=page, quarter=quarter))
    return sources


def answer_question(question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    collection = load_collection()
    api_client = load_api_client()
    quarter = detect_requested_quarter(question)
    question_embedding = embed_question(api_client, question)
    chunks = query_collection(collection, question_embedding, top_k=top_k, quarter=quarter)
    sources = build_sources(chunks)

    if not chunks:
        return {
            "answer": REFUSAL_TEXT,
            "sources": [],
            "quarter": quarter,
            "retrieved_chunks": [],
        }

    context = build_context(chunks)
    system_prompt = (
        "You are a financial report question-answering assistant.\n\n"
        "Answer ONLY using the provided context from the indexed HCLTech financial reports.\n"
        "Do not use outside knowledge.\n"
        "Do not invent financial values.\n"
        "Do not guess.\n"
        "Do not use information that is not present in the retrieved context.\n\n"
        f"If the requested information cannot be found in the provided context, respond exactly with: {REFUSAL_TEXT!r}.\n"
        "When reporting financial figures, include the appropriate unit and quarter/period whenever available."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context. If the answer is not in the context, refuse."
    )

    try:
        completion = api_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=512,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Gemini generation failed: {exc}") from exc

    answer = (completion.text or "").strip()
    if not answer:
        answer = REFUSAL_TEXT

    return {
        "answer": answer,
        "sources": [source.__dict__ for source in sources],
        "quarter": quarter,
        "retrieved_chunks": chunks,
    }


if __name__ == "__main__":
    sample = answer_question("What was the revenue in the latest quarter?")
    print(sample["answer"])
    for source in sample["sources"]:
        print(f"- {source['source']} | Page {source['page']} | {source['quarter']}")