from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from typing import Dict, Any
from datetime import datetime
from pydantic import BaseModel

from app.services.ml_service import get_ml_service
from app.services.chat_service import get_chat_service
from app.services.pdf_service import generate_report_pdf

router = APIRouter()


# =========================
# Request Models
# =========================

class PDFReportRequest(BaseModel):
    prediction_data: Dict[str, Any]
    patient_name: str = "Patient"


# =========================
# Routes
# =========================

@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)) -> Dict:
    """
    Upload an eye image and get disease prediction
    """

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpeg, png, etc.)"
        )

    try:
        # Read image bytes
        image_bytes = await file.read()

        # ML inference
        ml_service = get_ml_service()
        result = ml_service.predict(image_bytes)

        # Handle invalid/low confidence case
        if result.get("error"):
            return {
                "success": False,
                "error_type": "validation_failed",
                "message": result["message"],
                "filename": file.filename,
                "prediction": result
            }

        # AI doctor advice
        try:
            chat_service = get_chat_service()
            advice = chat_service.get_medical_advice(
                disease=result["predicted_class"],
                confidence=result["confidence"]
            )
        except Exception:
            advice = "AI advice temporarily unavailable."

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result,
            "ai_advice": advice
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict:
    """Check if prediction service is working"""
    try:
        ml_service = get_ml_service()
        return {
            "success": True,
            "message": "Prediction service is healthy",
            "model_loaded": ml_service.model is not None,
            "classes": ml_service.classes
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Service unhealthy: {str(e)}"
        )


@router.post("/generate-report")
async def generate_pdf_report(request: PDFReportRequest) -> Response:
    """Generate PDF report from prediction"""
    try:
        pdf_bytes = generate_report_pdf(
            request.prediction_data,
            request.patient_name
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=EyeTrust_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )