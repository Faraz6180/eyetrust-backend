from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from app.services.chat_service import get_chat_service

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict]] = None

class MedicalAdviceRequest(BaseModel):
    disease: str
    confidence: float
    question: Optional[str] = None

@router.post("/chat")
async def chat_with_doctor(request: ChatRequest) -> Dict:
    """
    Chat with AI doctor
    
    Example request:
    {
        "message": "What are the symptoms of cataract?",
        "conversation_history": []
    }
    """
    try:
        chat_service = get_chat_service()
        response = chat_service.chat(
            user_message=request.message,
            conversation_history=request.conversation_history
        )
        
        return {
            "success": True,
            "response": response
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

@router.post("/medical-advice")
async def get_medical_advice(request: MedicalAdviceRequest) -> Dict:
    """
    Get medical advice based on prediction
    
    Example request:
    {
        "disease": "Cataract",
        "confidence": 0.95,
        "question": "Is surgery necessary?"
    }
    """
    try:
        chat_service = get_chat_service()
        advice = chat_service.get_medical_advice(
            disease=request.disease,
            confidence=request.confidence,
            user_question=request.question
        )
        
        return {
            "success": True,
            "disease": request.disease,
            "confidence": request.confidence,
            "advice": advice
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate advice: {str(e)}"
        )