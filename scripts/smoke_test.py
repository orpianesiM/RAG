import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from ingest import ingest_documents
from rag import get_rag_response

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def main() -> None:
    ingest_documents()
    cases = [
        (
            "¿Qué función cumplen los embeddings en la búsqueda semántica?",
            False,
        ),
        (
            "¿Cuál es la contraseña de administrador del router del laboratorio?",
            True,
        ),
    ]

    for query, expected_abstention in cases:
        result = await get_rag_response(query)
        print(
            json.dumps(
                {
                    "consulta": query,
                    **result.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if result.sin_respuesta != expected_abstention:
            raise AssertionError(
                f"Resultado inesperado para la consulta: {query}"
            )
        if not expected_abstention and not result.referencias:
            raise AssertionError("La respuesta conocida debe tener referencias.")


if __name__ == "__main__":
    asyncio.run(main())
