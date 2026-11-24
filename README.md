# RAG local: recuperación semántica grounded

Proyecto de ejemplo End-to-End: carga documentos Markdown o TXT, los fragmenta
por tokens, indexa embeddings en ChromaDB persistente y responde consultas con
un LLM usando únicamente el contexto recuperado. Las respuestas y sus
referencias se validan con Pydantic.

## Requisitos

- Python 3.11 o superior
- Una clave válida de OpenAI para crear embeddings y consultar el modelo

## Instalación en Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Configura `OPENAI_API_KEY` dentro de `.env`. No subas ese archivo al
repositorio. El modelo de embeddings por defecto es
`text-embedding-3-small`; se usa el mismo al indexar y recuperar.

## Ingestar y persistir documentos

Coloca archivos `.md` o `.txt` en `data/` y ejecuta:

```powershell
python ingest.py
```

La ingesta utiliza `RecursiveCharacterTextSplitter` con un tamaño objetivo de
500 tokens y 50 tokens de solapamiento. Como en cualquier splitter, el último
fragmento de un archivo puede tener menos de 500 tokens. Los embeddings se
guardan en `vectorstore/`, que está excluido de Git.

El script calcula IDs estables para los fragmentos y evita re-embebir los que
no cambiaron. Si se edita o elimina un archivo, solo se agregan los fragmentos
nuevos y se eliminan los obsoletos. El manifiesto del índice registra la
colección y el modelo de embeddings; si no coincide con la configuración
actual, el programa falla explícitamente para evitar mezclar espacios
vectoriales. Para reconstruir el índice desde cero, elimina manualmente
`vectorstore/` y vuelve a correr la ingesta.

## Ejecutar dos consultas de prueba

```powershell
python scripts/smoke_test.py
```

El script consulta un dato presente en el corpus y espera referencias a sus
fragmentos. Después plantea una pregunta trampa cuya respuesta no está en los
documentos y exige que el modelo responda exactamente `No lo sé`, sin
referencias. Ambas consultas usan OpenAI y requieren una clave válida.

## Pruebas automatizadas sin API

```powershell
python -m pytest -q
```

Las pruebas usan una colección Chroma temporal, embeddings deterministas y un
modelo conversacional simulado; no hacen llamadas de red ni consumen créditos.
Cubren recuperación con fuente, abstención ante una pregunta trampa, rechazo
de referencias inventadas e ingesta idempotente.

## Uso desde Python

```python
import asyncio

from rag import get_rag_response

response = asyncio.run(
    get_rag_response("¿Qué función cumplen los embeddings?")
)
print(response.model_dump(mode="json"))
```

`ingest.py` implementa la carga y chunking; `vector_store.py` administra
ChromaDB persistente y comprueba que el modelo de embeddings sea consistente;
`rag.py` compone la cadena LCEL asíncrona con `PydanticOutputParser` y
`.ainvoke()`; `schemas.py` define la respuesta y las referencias. La
recuperación limita el contexto a `RAG_TOP_K` fragmentos (4 por defecto).

La salida estructurada incluye `respuesta`, `referencias` y `sin_respuesta`.
La aplicación verifica que cada referencia pertenezca a los fragmentos
recuperados; no acepta identificadores inventados.
