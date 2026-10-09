from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import io
from typing import List, Optional
from explainer import AdaptiveXAIEngine

app = FastAPI(
    title="Explainable AI Adaptive Education API",
    description="Enterprise API delivering student risk predictions, XAI attributions, one-click communication, and live CSV batch analysis.",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AdaptiveXAIEngine()

class StudentMetricsInput(BaseModel):
    student_id: Optional[str] = Field("STD-1001", json_schema_extra={"example": "STD-1001"})
    gender: str = Field("M", json_schema_extra={"example": "M"})
    highest_education: str = Field("HE Qualification", json_schema_extra={"example": "HE Qualification"})
    imd_band: str = Field("90-100%", json_schema_extra={"example": "90-100%"})
    age_band: str = Field("0-35", json_schema_extra={"example": "0-35"})
    disability: str = Field("N", json_schema_extra={"example": "N"})
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

class WhatIfRequest(BaseModel):
    student: StudentMetricsInput
    delta_vle_days: float = Field(15.0, json_schema_extra={"example": 15.0})
    delta_score: float = Field(10.0, json_schema_extra={"example": 10.0})

@app.get("/")
def read_root():
    return {
        "service": "XAI Adaptive Education Enterprise API",
        "status": "Healthy",
        "version": "2.2.0",
        "docs_url": "/docs"
    }

@app.post("/api/predict-and-explain")
def predict_and_explain(student: StudentMetricsInput):
    try:
        data_dict = student.model_dump()
        student_id = data_dict.pop("student_id", "STD-1001")
        input_df = pd.DataFrame([data_dict])
        explanation = engine.explain_student(input_df)
        explanation["student_id"] = student_id
        return {
            "status": "success",
            "data": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/generate-communication")
def generate_communication(student: StudentMetricsInput):
    try:
        data_dict = student.model_dump()
        student_id = data_dict.pop("student_id", "STD-1001")
        input_df = pd.DataFrame([data_dict])
        drafts = engine.generate_communication_draft(student_id, input_df)
        return {
            "status": "success",
            "data": drafts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Communication error: {str(e)}")

@app.post("/api/what-if-simulation")
def simulate_what_if(req: WhatIfRequest):
    try:
        data_dict = req.student.model_dump()
        data_dict.pop("student_id", None)
        input_df = pd.DataFrame([data_dict])
        simulation = engine.simulate_what_if(input_df, req.delta_vle_days, req.delta_score)
        return {
            "status": "success",
            "data": simulation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

@app.post("/api/batch-analyze")
def batch_analyze(students: List[StudentMetricsInput]):
    try:
        results = []
        at_risk_count = 0
        total_pass_prob = 0.0

        for s in students:
            data_dict = s.model_dump()
            sid = data_dict.pop("student_id", "STD-UNKNOWN")
            input_df = pd.DataFrame([data_dict])
            exp = engine.explain_student(input_df)
            exp["student_id"] = sid
            results.append(exp)
            
            if "At-Risk" in exp["predicted_status"]:
                at_risk_count += 1
            total_pass_prob += exp["pass_probability"]

        cohort_summary = {
            "total_students": len(students),
            "at_risk_count": at_risk_count,
            "on_track_count": len(students) - at_risk_count,
            "class_average_pass_prob": round(total_pass_prob / max(len(students), 1), 2)
        }

        return {
            "status": "success",
            "cohort_summary": cohort_summary,
            "students": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis error: {str(e)}")

@app.post("/api/upload-csv")
async def upload_csv_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        results = []
        at_risk_count = 0
        total_pass_prob = 0.0

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            student_id = str(row_dict.pop("student_id", f"STD-{1000 + idx}"))
            
            row_df = pd.DataFrame([row_dict])
            exp = engine.explain_student(row_df)
            exp["student_id"] = student_id
            results.append(exp)

            if "At-Risk" in exp["predicted_status"]:
                at_risk_count += 1
            total_pass_prob += exp["pass_probability"]

        cohort_summary = {
            "total_students": len(df),
            "at_risk_count": at_risk_count,
            "on_track_count": len(df) - at_risk_count,
            "class_average_pass_prob": round(total_pass_prob / max(len(df), 1), 2)
        }

        return {
            "status": "success",
            "filename": file.filename,
            "cohort_summary": cohort_summary,
            "students": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)