from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: NonEmptyText = Field(description="Ruta relativa del archivo de origen.")
    chunk_id: NonEmptyText = Field(description="Identificador exacto del fragmento recuperado.")


class RAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    respuesta: NonEmptyText = Field(
        description="Respuesta sustentada únicamente por los fragmentos recuperados; si no hay evidencia suficiente, debe ser exactamente 'No lo sé'."
    )
    referencias: list[SourceReference] = Field(
        description="Archivos y fragmentos recuperados que respaldan la respuesta; debe ser una lista vacía al responder 'No lo sé'."
    )
    sin_respuesta: bool = Field(
        description="Indica si el contexto no contiene evidencia suficiente para responder."
    )

    @model_validator(mode="after")
    def validate_abstention(self) -> "RAGResponse":
        if self.sin_respuesta:
            if self.respuesta != "No lo sé" or self.referencias:
                raise ValueError(
                    "Una respuesta sin evidencia debe ser exactamente 'No lo sé' y no incluir referencias."
                )
        else:
            if self.respuesta == "No lo sé":
                raise ValueError(
                    "Una respuesta 'No lo sé' debe indicar sin_respuesta=true."
                )
            if not self.referencias:
                raise ValueError(
                    "Una respuesta sustentada debe incluir al menos una referencia."
                )
        return self
