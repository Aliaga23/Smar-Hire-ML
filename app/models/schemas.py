from pydantic import BaseModel, Field
from typing import List, Literal

class Habilidad(BaseModel):
    nombre: str
    nivel: int = Field(..., ge=0, le=10, description="Nivel de dominio de 0 a 10")

class HabilidadVacante(Habilidad):
    requerido: Literal["SI", "NO"]

class Idioma(BaseModel):
    nombre: str
    nivel: int = Field(..., ge=0, le=10, description="Nivel de fluidez de 0 a 10")

class Vacante(BaseModel):
    id: str
    habilidades: List[HabilidadVacante]
    idiomas: List[Idioma]

class Candidato(BaseModel):
    id: str
    habilidades: List[Habilidad]
    idiomas: List[Idioma]
    experiencia_total_meses: int = Field(..., ge=0)

class MatchingRequest(BaseModel):
    vacante: Vacante
    candidatos: List[Candidato]

class DiferenciaHabilidad(BaseModel):
    nombre: str
    nivel_requerido: int
    nivel_candidato: int
    diferencia: int

class DiferenciaIdioma(BaseModel):
    nombre: str
    nivel_requerido: int
    nivel_candidato: int
    diferencia: int

class CandidatoRankeado(BaseModel):
    candidatoId: str
    score_compatibilidad: float = Field(..., ge=0, le=1)
    habilidades_diferencias: List[DiferenciaHabilidad]
    idiomas_diferencias: List[DiferenciaIdioma]

class MatchingResponse(BaseModel):
    candidatos_rankeados: List[CandidatoRankeado]
