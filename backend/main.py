from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from app.core.config import settings
from app.api.routes import predictions, chat

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)

# Add CORS middleware (allows frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    predictions.router,
    prefix="/api",
    tags=["predictions"]
)


app.include_router(
    chat.router,
    prefix="/api",
    tags=["chat"]
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to EyeTrust API",
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
