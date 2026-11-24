import asyncio
import hashlib
import json
import re
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.documents import Document

from ingest import ingest_documents
from rag import get_rag_response


class HashEmbeddings(Embeddings):
    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 256
        for token in re.findall(r"\w+", text.casefold()):
            index = int.from_bytes(
                hashlib.sha256(token.encode("utf-8")).digest()[:4],
                byteorder="big",
            ) % len(vector)
            vector[index] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _store(tmp_path: Path) -> Chroma:
    store = Chroma(
        collection_name="test_rag",
        embedding_function=HashEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    store.add_documents(
        [
            Document(
                page_content=(
                    "RAG combina la recuperación de fragmentos relevantes con un "
                    "modelo generativo. Los embeddings representan el significado "
                    "de textos como vectores para buscar por similitud semántica."
                ),
                metadata={"source": "conceptos.md", "chunk_id": "chunk-rag"},
            ),
            Document(
                page_content=(
                    "ChromaDB almacena vectores y metadatos de documentos en una "
                    "colección persistente para recuperarlos posteriormente."
                ),
                metadata={"source": "chroma.md", "chunk_id": "chunk-chroma"},
            ),
        ],
        ids=["chunk-rag", "chunk-chroma"],
    )
    return store


def test_answers_from_retrieved_context_with_reference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    llm = FakeListChatModel(
        responses=[
            json.dumps(
                {
                    "respuesta": "RAG recupera fragmentos relevantes y los usa como contexto para generar la respuesta.",
                    "referencias": [
                        {"source": "conceptos.md", "chunk_id": "chunk-rag"}
                    ],
                    "sin_respuesta": False,
                },
                ensure_ascii=False,
            )
        ]
    )

    result = asyncio.run(
        get_rag_response(
            "¿Qué hace RAG con los fragmentos recuperados?",
            vector_store=store,
            llm=llm,
        )
    )

    assert not result.sin_respuesta
    assert result.referencias[0].source == "conceptos.md"
    assert "fragmentos relevantes" in result.respuesta


def test_abstains_when_trick_question_is_not_in_context(tmp_path: Path) -> None:
    store = _store(tmp_path)
    llm = FakeListChatModel(
        responses=[
            json.dumps(
                {
                    "respuesta": "No lo sé",
                    "referencias": [],
                    "sin_respuesta": True,
                },
                ensure_ascii=False,
            )
        ]
    )

    result = asyncio.run(
        get_rag_response(
            "¿Cuál es la contraseña del router wifi del laboratorio?",
            vector_store=store,
            llm=llm,
        )
    )

    assert result.respuesta == "No lo sé"
    assert result.referencias == []
    assert result.sin_respuesta


def test_rejects_reference_not_returned_by_retriever(tmp_path: Path) -> None:
    store = _store(tmp_path)
    llm = FakeListChatModel(
        responses=[
            json.dumps(
                {
                    "respuesta": "Una respuesta sin sustento.",
                    "referencias": [
                        {"source": "inventado.md", "chunk_id": "fake"}
                    ],
                    "sin_respuesta": False,
                },
                ensure_ascii=False,
            )
        ]
    )

    with pytest.raises(ValueError, match="referencia no recuperada"):
        asyncio.run(
            get_rag_response(
                "¿Qué hace RAG?",
                vector_store=store,
                llm=llm,
            )
        )


def test_retries_after_invalid_model_output(tmp_path: Path) -> None:
    store = _store(tmp_path)
    llm = FakeListChatModel(
        responses=[
            "Esto no es JSON",
            json.dumps(
                {
                    "respuesta": "RAG recupera fragmentos relevantes para responder.",
                    "referencias": [
                        {"source": "conceptos.md", "chunk_id": "chunk-rag"}
                    ],
                    "sin_respuesta": False,
                },
                ensure_ascii=False,
            ),
        ]
    )

    result = asyncio.run(
        get_rag_response(
            "¿Qué hace RAG con los fragmentos?",
            vector_store=store,
            llm=llm,
        )
    )

    assert not result.sin_respuesta
    assert result.referencias[0].chunk_id == "chunk-rag"


def test_ingestion_is_idempotent_and_uses_configured_chunks(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    source = data_directory / "manual.md"
    paragraph = (
        "Los embeddings representan el significado de los textos como vectores. "
        "La búsqueda semántica compara la consulta con los documentos almacenados. "
    )
    source.write_text(paragraph * 160, encoding="utf-8")
    persist_directory = tmp_path / "vectorstore"
    embeddings = HashEmbeddings()

    first_added = ingest_documents(
        data_directory=data_directory,
        persist_directory=persist_directory,
        embeddings=embeddings,
    )
    second_added = ingest_documents(
        data_directory=data_directory,
        persist_directory=persist_directory,
        embeddings=embeddings,
    )
    stored = Chroma(
        collection_name="documentos_rag",
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    assert first_added > 1
    assert second_added == 0
    assert len(stored.get(include=[])["ids"]) == first_added
