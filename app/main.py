from fastapi import FastAPI
from app.core.config import settings
from app.core.lifespan import lifespan
from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.qdrant import router as qdrant_router
from app.api.routes.query import router as query_router
from app.api.routes.research import router as research_router

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(qdrant_router)
app.include_router(query_router)
app.include_router(research_router)
