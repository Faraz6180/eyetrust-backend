from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "EyeTrust API"
    APP_VERSION: str = "1.0.0"
    
    # Model Configuration
    MODEL_PATH: str = "models/eyetrust_mobilenetv2.h5"
    MODEL_CONFIG_PATH: str = "models/model_config.json"
    INPUT_SIZE: int = 224
    
    # Disease Classes
    CLASSES: List[str] = [
        "Cataract",
        "Conjunctivitis", 
        "Eyelid",
        "Normal",
        "Uveitis"
    ]
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "*",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
    ]

    # Prediction threshold for unknown/out-of-distribution images
    PREDICTION_THRESHOLD: float = 0.5
    
    class Config:
        case_sensitive = True

# Create settings instance
settings = Settings()
