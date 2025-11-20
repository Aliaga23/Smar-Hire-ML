from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Sistema de Matching - Candidatos y Vacantes",
    description="Microservicio de ML para calcular compatibilidad entre candidatos y vacantes usando K-means",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(router)

@app.get("/")
async def root():
    return {
        "message": "Sistema de Matching ML - Candidatos y Vacantes",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
