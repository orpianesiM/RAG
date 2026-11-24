import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = Path(
    os.getenv("CHROMA_PERSIST_DIRECTORY", str(PROJECT_DIR / "vectorstore"))
)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documentos_rag")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
MANIFEST_NAME = "rag_manifest.json"


def _manifest_path(persist_directory: Path) -> Path:
    return persist_directory / MANIFEST_NAME


def _read_manifest(persist_directory: Path) -> dict[str, Any] | None:
    manifest_path = _manifest_path(persist_directory)
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"No se pudo leer el manifiesto RAG: {manifest_path}") from error

    expected = {
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "La configuración del índice persistido no coincide con la actual: "
            + ", ".join(mismatches)
            + ". Usa la misma colección y el mismo modelo de embeddings o elimina "
            "el directorio vectorstore para reconstruirlo."
        )
    return manifest


def write_manifest(persist_directory: Path) -> None:
    manifest_path = _manifest_path(persist_directory)
    manifest_path.write_text(
        json.dumps(
            {
                "collection_name": COLLECTION_NAME,
                "embedding_model": EMBEDDING_MODEL,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_vector_store(
    *,
    persist_directory: str | Path = VECTOR_STORE_DIR,
    embeddings: Embeddings | None = None,
) -> Chroma:
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    _read_manifest(directory)
    embedding_function = embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(directory),
    )


def load_vector_store(
    *,
    persist_directory: str | Path = VECTOR_STORE_DIR,
    embeddings: Embeddings | None = None,
) -> Chroma:
    directory = Path(persist_directory)
    if _read_manifest(directory) is None:
        raise FileNotFoundError(
            f"No se encontró un índice RAG en {directory}. Ejecuta `python ingest.py` primero."
        )
    return create_vector_store(
        persist_directory=directory,
        embeddings=embeddings,
    )
