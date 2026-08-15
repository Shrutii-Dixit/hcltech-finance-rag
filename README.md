# FinSight – HCLTech Financial Intelligence Assistant

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) application for answering questions from HCLTech quarterly financial report PDFs. It follows the Assignment 1 step-by-step guide and uses Google Gemini as the allowed alternative AI provider because OpenAI API quota was unavailable in this workspace. The system is still designed to stay grounded in the uploaded company reports rather than using general model knowledge.

## Problem / Objective

The goal is to index four consecutive quarterly HCLTech financial reports, retrieve the most relevant passages by meaning, and generate answers with Gemini while showing the filename, page number, and quarter for each source used.

## Company

HCLTech

## Documents Used

The application reads the PDFs dynamically from the local `data/` folder:

- `HCLTech_Q1_FY26.pdf`
- `HCLTech_Q2_FY26.pdf`
- `HCLTech_Q3_FY26.pdf`
- `HCLTech_Q4_FY26.pdf`

## Official HCLTech Financial Reports

The project uses HCLTech's official quarterly Investor Releases for FY26:

- [Q1 FY26 – Investor Release](https://www.hcltech.com/sites/default/files/document/open/quarter-results/2025-07/hcltech-q1-fy26-investor-release.pdf)
- [Q2 FY26 – Investor Release](https://www.hcltech.com/sites/default/files/document/open/quarter-results/2025-11/HCLTech_Q2-FY26-Investor.pdf)
- [Q3 FY26 – Investor Release](https://www.hcltech.com/sites/default/files/document/open/quarter-results/2026-01/HCLTech_Q3_FY26_Investor_Release.pdf)
- [Q4 FY26 – Investor Release](https://www.hcltech.com/investor-relations/quarter-results)

Source: [HCLTech Investor Relations – Quarter Results](https://www.hcltech.com/investor-relations/quarter-results)

## Architecture / Workflow

PDF reports -> PDF text extraction -> page-level metadata -> recursive chunking -> source-prefixed chunk text -> Gemini embeddings -> persistent ChromaDB -> user question -> the same embedding model -> semantic retrieval -> quarter-aware filtering when possible -> Gemini -> grounded answer -> sources with filename, page number, and quarter.

## Technologies

- Python
- pypdf
- Google Gen AI SDK
- `gemini-embedding-2`
- `gemini-2.5-flash`
- ChromaDB
- Streamlit
- python-dotenv
- LangChain text splitters

## Project Structure

```text
HCLTech-Finance-RAG/
├── data/
│   ├── HCLTech_Q1_FY26.pdf
│   ├── HCLTech_Q2_FY26.pdf
│   ├── HCLTech_Q3_FY26.pdf
│   └── HCLTech_Q4_FY26.pdf
├── chroma_db/
├── ingest.py
├── rag.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

## Setup Instructions

1. Create and activate a Python virtual environment for this project.
2. Install the dependencies from `requirements.txt`.
3. Put your Gemini API key in `.env` as `GEMINI_API_KEY=...`.
4. Make sure the HCLTech PDFs are present in `data/`.

## Virtual Environment Setup

Example for Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment Variable Setup

Create or update the `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
```

Do not commit the key and do not place it in screenshots.

## How to Add PDFs

Place the quarterly HCLTech PDFs in the `data/` folder. The Streamlit app also lets you upload PDFs and save them into `data/` before indexing.

## How to Index Documents

Run the ingestion script:

```powershell
python ingest.py
```

This reads every PDF in `data/`, extracts page text, chunks it, creates embeddings with `gemini-embedding-2`, and stores the vectors in persistent ChromaDB under `chroma_db/`.

## How to Run Streamlit

```powershell
streamlit run app.py
```

Use the UI in this order: upload -> index -> ask -> answer -> sources.

## Chunking Configuration

- Chunk size: `1200`
- Chunk overlap: `150`

### Reason for the Chunking Choice

Financial reports often contain tables and compact figure-heavy sections. The upper end of the allowed chunk size helps keep table context together, while the overlap reduces the chance that useful figures or headings are split across chunk boundaries.

## Embedding Model

- `gemini-embedding-2`

The same embedding model is used for both the documents and the user question.

## Generation Model

- `gemini-2.5-flash`

The generation prompt is strict and grounded: it answers only from retrieved context and refuses unsupported questions.

## ChromaDB Persistence

ChromaDB is configured with a persistent directory at `./chroma_db`. The ingestion flow rebuilds the collection cleanly so duplicate chunks are not created on repeated runs.

## Quarter-Aware Retrieval

Each chunk is prefixed before embedding with source information like filename, quarter, and page. This makes the quarter part of the semantic search input, not just metadata. The retriever also detects an explicitly mentioned quarter in the user question and prioritizes matching chunks when possible.

## Source Attribution

Every answer shows the supporting sources dynamically from ChromaDB metadata, including:

- filename
- page number
- quarter

## Unsupported Question Handling

If the requested information is not present in the provided reports, the system refuses with:

`I could not find this information in the provided reports.`

This includes trap questions about companies or facts not contained in the HCLTech PDFs.

## Testing

The assignment guide asks for the following ten question categories. Fill in the results after you test the app yourself.

| # | Question | Answer | Correct? | Notes |
|---|---|---|---|---|
| 1 | Revenue in the latest quarter |  |  |  |
| 2 | Net profit compared across quarters |  |  |  |
| 3 | Year-on-year revenue comparison |  |  |  |
| 4 | Management commentary on demand |  |  |  |
| 5 | Fastest-growing segment |  |  |  |
| 6 | Operating margin trend |  |  |  |
| 7 | Dividend declared |  |  |  |
| 8 | Risks and headwinds |  |  |  |
| 9 | Three-line summary |  |  |  |
| 10 | Trap question that must be refused |  |  |  |

## Limitations

- Answer quality depends on PDF text extraction quality.
- Scanned or image-only PDFs will not produce usable text with standard extraction.
- The app is designed for the HCLTech quarterly report set in this workspace.

## Future Improvements

- Add more robust table extraction for dense financial pages.
- Add cached indexing status reporting in the UI.
- Add an optional backend service if the assignment scope expands.

## Notes

Do not claim test success until the application has actually been run on the target PDFs.
