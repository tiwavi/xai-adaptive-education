from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from explainer import AdaptiveXAIEngine

app = FastAPI(
    title="Explainable AI Adaptive Education API",
    description="Backend API delivering real-time student performance predictions and personalized XAI explanations.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize XAI Engine instance
engine = AdaptiveXAIEngine()

# Pydantic V2 Schema for Input Validation
class StudentMetricsInput(BaseModel):
    gender: str = Field(..., json_schema_extra={"example": "M"})
    highest_education: str = Field(..., json_schema_extra={"example": "HE Qualification"})
    imd_band: str = Field(..., json_schema_extra={"example": "90-100%"})
    age_band: str = Field(..., json_schema_extra={"example": "0-35"})
    disability: str = Field(..., json_schema_extra={"example": "N"})
    num_of_prev_attempts: int = Field(0, ge=0, json_schema_extra={"example": 0})
    studied_credits: int = Field(60, ge=0, json_schema_extra={"example": 60})
    module_presentation_length: int = Field(268, json_schema_extra={"example": 268})
    date_registration: float = Field(-50.0, json_schema_extra={"example": -50.0})
    avg_assessment_score: float = Field(75.0, ge=0.0, le=100.0, json_schema_extra={"example": 75.0})
    total_weighted_score: float = Field(70.0, ge=0.0, json_schema_extra={"example": 70.0})
    assessments_completed: float = Field(5.0, ge=0.0, json_schema_extra={"example": 5.0})
    avg_submission_delay: float = Field(0.0, json_schema_extra={"example": 0.0})
    total_clicks: float = Field(1200.0, ge=0.0, json_schema_extra={"example": 1200.0})
    mean_daily_clicks: float = Field(3.5, ge=0.0, json_schema_extra={"example": 3.5})
    active_vle_days: float = Field(50.0, ge=0.0, json_schema_extra={"example": 50.0})
    early_clicks: float = Field(80.0, ge=0.0, json_schema_extra={"example": 80.0})

@app.get("/")
def read_root():
    return {
        "service": "XAI Adaptive Education API",
        "status": "Healthy",
        "docs_url": "/docs"
    }

@app.post("/api/predict-and-explain")
def predict_and_explain(student: StudentMetricsInput):
    try:
        data_dict = student.model_dump()
        input_df = pd.DataFrame([data_dict])
        explanation = engine.explain_student(input_df)
        return {
            "status": "success",
            "data": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)