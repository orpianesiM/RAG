import hashlib
import logging
import os
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import VECTOR_STORE_DIR, create_vector_store, write_manifest

load_dotenv()

logger = logging.getLogger(__name__)
DATA_DIR = Path(os.getenv("RAG_DATA_DIRECTORY", str(Path(__file__).resolve().parent / "data")))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
TOKEN_ENCODING = os.getenv("RAG_TOKEN_ENCODING", "cl100k_base")


def _load_documents(data_directory: Path) -> list[Document]:
    if not data_directory.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de documentos: {data_directory}")

    documents: list[Document] = []
    for path in sorted((*data_directory.glob("*.txt"), *data_directory.glob("*.md"))):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            logger.warning("Se omite el documento vacío: %s", path.name)
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={"source": path.relative_to(data_directory).as_posix()},
            )
        )

    if not documents:
        raise ValueError(
            f"No hay archivos .txt o .md con contenido para ingerir en {data_directory}."
        )
    return documents


def _build_chunks(documents: list[Document]) -> list[Document]:
    if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "La configuración de chunking requiere chunk_size > chunk_overlap >= 0."
        )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKEN_ENCODING,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    tokenizer = tiktoken.get_encoding(TOKEN_ENCODING)

    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata["source"])
        start_index = int(chunk.metadata.get("start_index", 0))
        token_count = len(tokenizer.encode(chunk.page_content))
        chunk_id = hashlib.sha256(
            f"{source}:{start_index}:{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        chunk.metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_index": index,
                "token_count": token_count,
            }
        )
    return chunks


def ingest_documents(
    *,
    data_directory: str | Path = DATA_DIR,
    persist_directory: str | Path = VECTOR_STORE_DIR,
    embeddings: Embeddings | None = None,
) -> int:
    """Index files idempotently; return the number of newly embedded chunks."""
    data_path = Path(data_directory)
    persist_path = Path(persist_directory)
    documents = _load_documents(data_path)
    chunks = _build_chunks(documents)
    store = create_vector_store(
        persist_directory=persist_path,
        embeddings=embeddings,
    )

    existing_ids = set(store.get(include=[])["ids"])
    desired_ids = {str(chunk.metadata["chunk_id"]) for chunk in chunks}
    stale_ids = existing_ids - desired_ids
    missing_chunks = [
        chunk for chunk in chunks if str(chunk.metadata["chunk_id"]) not in existing_ids
    ]

    if missing_chunks:
        store.add_documents(
            documents=missing_chunks,
            ids=[str(chunk.metadata["chunk_id"]) for chunk in missing_chunks],
        )
    if stale_ids:
        store.delete(ids=sorted(stale_ids))

    write_manifest(persist_path)
    logger.info(
        "Ingesta completa: %d fragmentos totales, %d nuevos y %d obsoletos eliminados.",
        len(chunks),
        len(missing_chunks),
        len(stale_ids),
    )
    return len(missing_chunks)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    newly_added = ingest_documents()
    print(f"Ingesta completa. Fragmentos nuevos: {newly_added}")
