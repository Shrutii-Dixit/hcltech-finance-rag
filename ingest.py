from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
from pypdf import PdfReader


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "hcltech_finance_reports"
EMBEDDING_MODEL = "gemini-embedding-2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
BATCH_SIZE = 20
MAX_RETRIES_PER_BATCH = 3
RETRY_WAIT_SECONDS = 65


@dataclass
class PageRecord:
    source: str
    page: int
    quarter: str
    text: str


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def load_gemini_client() -> genai.Client:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file in the project root.")
    return genai.Client(api_key=api_key)


def load_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def quarter_from_filename(filename: str) -> str:
    match = re.search(r"(Q[1-4])[_\-\s]*FY(\d{2,4})", filename, re.IGNORECASE)
    if not match:
        return "Unknown Quarter"
    quarter = match.group(1).upper()
    fy_value = match.group(2)
    if len(fy_value) == 4:
        fy_value = fy_value[-2:]
    return f"{quarter} FY{fy_value}"


def list_pdf_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def extract_pages(pdf_path: Path) -> tuple[list[PageRecord], int, list[int]]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not open PDF '{pdf_path.name}': {exc}") from exc

    quarter = quarter_from_filename(pdf_path.name)
    page_records: list[PageRecord] = []
    empty_pages: list[int] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            extracted_text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            extracted_text = ""
        if extracted_text.strip():
            page_records.append(
                PageRecord(
                    source=pdf_path.name,
                    page=page_number,
                    quarter=quarter,
                    text=extracted_text.strip(),
                )
            )
        else:
            empty_pages.append(page_number)

    return page_records, len(reader.pages), empty_pages


def build_chunks(page_records: list[PageRecord]) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunk_records: list[ChunkRecord] = []

    for record in page_records:
        prefix = f"[Source: {record.source} | Quarter: {record.quarter} | Page: {record.page}]"
        body_chunks = splitter.split_text(record.text)
        for chunk_index, body_chunk in enumerate(body_chunks):
            chunk_text = f"{prefix}\n\n{body_chunk}"
            chunk_id = f"{Path(record.source).stem.lower()}::p{record.page:03d}::c{chunk_index:03d}"
            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "source": record.source,
                        "page": record.page,
                        "quarter": record.quarter,
                        "chunk_index": chunk_index,
                    },
                )
            )

    return chunk_records


def embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )
        for text in texts
    ]

    for retry_index in range(MAX_RETRIES_PER_BATCH + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
            )
            embeddings = [embedding.values for embedding in response.embeddings]
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Expected {len(texts)} embeddings but received {len(embeddings)}"
                )
            return embeddings
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            is_quota_error = "429" in message or "RESOURCE_EXHAUSTED" in message
            if is_quota_error and retry_index < MAX_RETRIES_PER_BATCH:
                print(
                    f"Embedding rate limit hit (429). Waiting {RETRY_WAIT_SECONDS} seconds before retry "
                    f"{retry_index + 1}/{MAX_RETRIES_PER_BATCH}..."
                )
                time.sleep(RETRY_WAIT_SECONDS)
                continue
            raise

    raise RuntimeError("Embedding failed after retry attempts.")


def prepare_collection(client: chromadb.PersistentClient):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        pass
    return client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def collection_exists() -> bool:
    client = load_chroma_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        return False
    try:
        return collection.count() > 0
    except Exception:  # noqa: BLE001
        return False


def ingest_documents(data_dir: Path | None = None, rebuild: bool = True) -> dict[str, Any]:
    load_dotenv(ROOT_DIR / ".env")
    data_path = data_dir or DATA_DIR
    if not data_path.exists():
        raise RuntimeError(f"Data directory not found: {data_path}")

    pdf_files = list_pdf_files(data_path)
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {data_path}")

    gemini_client = load_gemini_client()
    chroma_client = load_chroma_client()
    collection = prepare_collection(chroma_client) if rebuild else chroma_client.get_or_create_collection(COLLECTION_NAME)

    all_page_records: list[PageRecord] = []
    total_pages = 0
    empty_pages_report: list[str] = []

    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Persistence directory: {CHROMA_DIR}")

    for pdf_path in pdf_files:
        page_records, page_count, empty_pages = extract_pages(pdf_path)
        total_pages += page_count
        all_page_records.extend(page_records)
        sample_text = page_records[0].text[:300].replace("\n", " ") if page_records else ""
        print(f"File: {pdf_path.name} | Pages: {page_count} | Extracted pages: {len(page_records)}")
        if sample_text:
            print(f"First page sample: {sample_text}")
        else:
            print("First page sample: <no selectable text found>")
        if empty_pages:
            empty_pages_report.append(f"{pdf_path.name}: pages {', '.join(str(page) for page in empty_pages)}")

    chunk_records = build_chunks(all_page_records)
    total_chunks = len(chunk_records)
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE if total_chunks else 0

    for batch_index, start_index in enumerate(range(0, total_chunks, BATCH_SIZE), start=1):
        print(f"Embedding batch {batch_index}/{total_batches}...")
        batch = chunk_records[start_index : start_index + BATCH_SIZE]
        embeddings = embed_texts(gemini_client, [record.text for record in batch])
        collection.add(
            ids=[record.chunk_id for record in batch],
            embeddings=embeddings,
            documents=[record.text for record in batch],
            metadatas=[record.metadata for record in batch],
        )

    print(f"Files processed: {len(pdf_files)}")
    print(f"Pages processed: {total_pages}")
    print(f"Chunks created: {total_chunks}")
    print(f"Collection chunk count: {collection.count()}")

    if empty_pages_report:
        print("Empty pages reported:")
        for line in empty_pages_report:
            print(f"- {line}")

    return {
        "files_processed": len(pdf_files),
        "pages_processed": total_pages,
        "chunks_created": total_chunks,
        "collection_name": COLLECTION_NAME,
        "persistence_directory": str(CHROMA_DIR),
        "collection_count": collection.count(),
        "empty_pages_report": empty_pages_report,
    }


def main() -> None:
    stats = ingest_documents()
    print("Ingestion complete.")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()