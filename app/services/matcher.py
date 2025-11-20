import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
from app.models.schemas import (
    MatchingRequest, MatchingResponse, CandidatoRankeado,
    DiferenciaHabilidad, DiferenciaIdioma, Candidato, Vacante
)

class CandidateVacancyMatcher:
    
    def __init__(
        self,
        peso_habilidades_obligatorias: float = 0.6,
        peso_habilidades_deseables: float = 0.25,
        peso_idiomas: float = 0.1,
        peso_experiencia: float = 0.05
    ):
        self.scaler = StandardScaler()
        self.peso_habilidades_obligatorias = peso_habilidades_obligatorias
        self.peso_habilidades_deseables = peso_habilidades_deseables
        self.peso_idiomas = peso_idiomas
        self.peso_experiencia = peso_experiencia
    
    def calculate_compatibility(self, request: MatchingRequest) -> MatchingResponse:
        vacante = request.vacante
        candidatos = request.candidatos
        
        max_experiencia = max(
            (c.experiencia_total_meses for c in candidatos),
            default=1
        )
        
        candidatos_rankeados = []
        
        for candidato in candidatos:
            habilidades_diffs = self._calculate_skill_differences(
                vacante.habilidades, 
                candidato.habilidades
            )
            
            idiomas_diffs = self._calculate_language_differences(
                vacante.idiomas, 
                candidato.idiomas
            )
            
            score = self._calculate_compatibility_score(
                vacante, 
                candidato, 
                habilidades_diffs, 
                idiomas_diffs,
                max_experiencia
            )
            
            candidatos_rankeados.append(
                CandidatoRankeado(
                    candidatoId=candidato.id,
                    score_compatibilidad=round(score, 2),
                    habilidades_diferencias=habilidades_diffs,
                    idiomas_diferencias=idiomas_diffs
                )
            )
        
        candidatos_rankeados.sort(
            key=lambda x: x.score_compatibilidad, 
            reverse=True
        )
        
        if len(candidatos) >= 2:
            self._apply_kmeans_clustering(candidatos, vacante)
        
        return MatchingResponse(candidatos_rankeados=candidatos_rankeados)
    
    def _calculate_skill_differences(
        self, 
        habilidades_vacante: List, 
        habilidades_candidato: List
    ) -> List[DiferenciaHabilidad]:
        diferencias = []
        candidato_skills_map = {
            h.nombre: h.nivel for h in habilidades_candidato
        }
        
        for habilidad_req in habilidades_vacante:
            nivel_candidato = candidato_skills_map.get(habilidad_req.nombre, 0)
            diferencia = nivel_candidato - habilidad_req.nivel
            
            diferencias.append(
                DiferenciaHabilidad(
                    nombre=habilidad_req.nombre,
                    nivel_requerido=habilidad_req.nivel,
                    nivel_candidato=nivel_candidato,
                    diferencia=diferencia
                )
            )
        
        return diferencias
    
    def _calculate_language_differences(
        self, 
        idiomas_vacante: List, 
        idiomas_candidato: List
    ) -> List[DiferenciaIdioma]:
        diferencias = []
        candidato_langs_map = {
            i.nombre: i.nivel for i in idiomas_candidato
        }
        
        for idioma_req in idiomas_vacante:
            nivel_candidato = candidato_langs_map.get(idioma_req.nombre, 0)
            diferencia = nivel_candidato - idioma_req.nivel
            
            diferencias.append(
                DiferenciaIdioma(
                    nombre=idioma_req.nombre,
                    nivel_requerido=idioma_req.nivel,
                    nivel_candidato=nivel_candidato,
                    diferencia=diferencia
                )
            )
        
        return diferencias
    
    def _calculate_compatibility_score(
        self,
        vacante: Vacante,
        candidato: Candidato,
        habilidades_diffs: List[DiferenciaHabilidad],
        idiomas_diffs: List[DiferenciaIdioma],
        max_experiencia: int
    ) -> float:
        habilidades_obligatorias = [
            h for h in vacante.habilidades if h.requerido == "SI"
        ]
        score_obligatorias = self._calculate_skill_score(
            habilidades_obligatorias, 
            habilidades_diffs
        )
        
        habilidades_deseables = [
            h for h in vacante.habilidades if h.requerido == "NO"
        ]
        score_deseables = self._calculate_skill_score(
            habilidades_deseables, 
            habilidades_diffs
        )
        
        score_idiomas = self._calculate_language_score(
            vacante.idiomas, 
            idiomas_diffs
        )
        
        score_experiencia = (
            candidato.experiencia_total_meses / max_experiencia 
            if max_experiencia > 0 else 0.0
        )
        
        score_total = (
            score_obligatorias * self.peso_habilidades_obligatorias +
            score_deseables * self.peso_habilidades_deseables +
            score_idiomas * self.peso_idiomas +
            score_experiencia * self.peso_experiencia
        )
        
        return max(0.0, min(1.0, score_total))
    
    def _calculate_skill_score(
        self, 
        habilidades_requeridas: List, 
        diferencias: List[DiferenciaHabilidad]
    ) -> float:
        if not habilidades_requeridas:
            return 1.0
        
        diffs_map = {d.nombre: d for d in diferencias}
        total_score = 0.0
        
        for hab_req in habilidades_requeridas:
            diff = diffs_map.get(hab_req.nombre)
            if diff:
                if diff.nivel_candidato >= diff.nivel_requerido:
                    total_score += 1.0
                elif diff.nivel_candidato == 0:
                    total_score += 0.0
                else:
                    ratio = diff.nivel_candidato / max(diff.nivel_requerido, 1)
                    total_score += ratio * 0.8
        
        return total_score / len(habilidades_requeridas)
    
    def _calculate_language_score(
        self, 
        idiomas_requeridos: List, 
        diferencias: List[DiferenciaIdioma]
    ) -> float:
        if not idiomas_requeridos:
            return 1.0
        
        diffs_map = {d.nombre: d for d in diferencias}
        total_score = 0.0
        
        for idioma_req in idiomas_requeridos:
            diff = diffs_map.get(idioma_req.nombre)
            if diff:
                if diff.nivel_candidato >= diff.nivel_requerido:
                    total_score += 1.0
                elif diff.nivel_candidato == 0:
                    total_score += 0.2
                else:
                    ratio = diff.nivel_candidato / max(diff.nivel_requerido, 1)
                    total_score += max(0.2, ratio)
        
        return total_score / len(idiomas_requeridos)
    
    def _apply_kmeans_clustering(
        self, 
        candidatos: List[Candidato], 
        vacante: Vacante
    ) -> Dict:
        if len(candidatos) < 2:
            return {}
        
        features = []
        for candidato in candidatos:
            avg_skills = np.mean([h.nivel for h in candidato.habilidades]) if candidato.habilidades else 0
            avg_langs = np.mean([i.nivel for i in candidato.idiomas]) if candidato.idiomas else 0
            exp = candidato.experiencia_total_meses
            
            features.append([avg_skills, avg_langs, exp])
        
        features_array = np.array(features)
        features_scaled = self.scaler.fit_transform(features_array)
        
        n_clusters = min(3, len(candidatos))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        return {
            "clusters": clusters.tolist(),
            "centroids": kmeans.cluster_centers_.tolist()
        }
