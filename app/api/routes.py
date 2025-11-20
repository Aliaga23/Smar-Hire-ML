from fastapi import APIRouter
from app.models.schemas import MatchingRequest, MatchingResponse
from app.services.matcher import CandidateVacancyMatcher

router = APIRouter(prefix="/api", tags=["matching"])

matcher = CandidateVacancyMatcher()

@router.post("/matching", response_model=MatchingResponse)
async def calculate_matching(request: MatchingRequest):
    """
    Calcula la compatibilidad entre candidatos y una vacante.
    
    Utiliza K-means y scoring basado en:
    - Habilidades (nivel de dominio)
    - Idiomas (nivel de fluidez)
    - Experiencia total en meses
    """
    result = matcher.calculate_compatibility(request)
    return result
