from fastapi import FastAPI
from app.core.config import settings
from app.core.lifespan import lifespan
from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.include_router(health_router)
app.include_router(ingest_router)