import logging
from contextlib import asynccontextmanager
from pathlib import Path

from backend.services.cmapss_loader import load_train

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database import engine, Base
import backend.models  # noqa: F401 — registers all ORM models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting AdaptiveML Monitor...")

        # Create DB tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified.")

        # Ensure required directories exist
        Path(settings.MODEL_STORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.CMAPSS_DATA_PATH).mkdir(parents=True, exist_ok=True)
        logger.info("Storage directories verified.")

        # Verify CMAPSS data is present and readable
        cmapss_files = list(Path(settings.CMAPSS_DATA_PATH).glob("*.txt"))
        if not cmapss_files:
            logger.warning(
                f"No CMAPSS .txt files found in {settings.CMAPSS_DATA_PATH}. "
                "Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"
            )
        else:
            logger.info(f"Found {len(cmapss_files)} CMAPSS files.")
            try:
                df = load_train("FD001")
                logger.info(
                    f"CMAPSS FD001 validated: {len(df)} training rows loaded."
                )
            except Exception as e:
                logger.error(f"CMAPSS validation failed: {e}")

        yield
        logger.info("Shutting down AdaptiveML Monitor.")

    application = FastAPI(
        title="AdaptiveML Monitor",
        description="Continual learning under delayed and incomplete labels.",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    return application


app = create_app()


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "env": settings.APP_ENV}