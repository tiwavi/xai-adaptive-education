import joblib
import pandas as pd
import numpy as np

class AdaptiveXAIEngine:
    """
    Explainable AI (XAI) Engine for Personalized Education.
    Computes local feature importance and generates dynamic educational interventions.
    """
    def __init__(self, model_path='model.pkl', explainer_path='explainer_model.pkl', 
                 preprocessor_path='preprocessor.pkl', features_path='transformed_feature_names.pkl'):
        self.model = joblib.load(model_path)
        self.explainer_model = joblib.load(explainer_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_names = joblib.load(features_path)
        
    def explain_student(self, student_df: pd.DataFrame) -> dict:
        student_df = student_df.copy()
        
        # 1. On-the-fly Domain Feature Engineering
        student_df['clicks_per_active_day'] = student_df['total_clicks'] / (student_df['active_vle_days'] + 1e-5)
        student_df['assessment_completion_ratio'] = student_df['assessments_completed'] / 14.0
        student_df['early_engagement_ratio'] = student_df['early_clicks'] / (student_df['total_clicks'] + 1.0)
        
        # 2. Transform input using pipeline
        X_proc = self.preprocessor.transform(student_df)
        
        # 3. Model Inference
        prob_pass = float(self.model.predict_proba(X_proc)[0][1])
        prediction = int(self.model.predict(X_proc)[0])
        
        # 4. Extract Feature Importances
        importances = self.explainer_model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:5]
        
        top_factors = []
        for idx in top_indices:
            clean_name = self.feature_names[idx].replace("num__", "").replace("cat__", "")
            top_factors.append({
                "feature": clean_name,
                "importance_percentage": round(float(importances[idx]) * 100, 2)
            })
            
        # 5. Rule-Based Intervention Rules Engine
        row = student_df.iloc[0]
        interventions = []
        
        if row['active_vle_days'] < 25:
            interventions.append({
                "type": "Engagement Nudge",
                "severity": "High" if row['active_vle_days'] < 10 else "Medium",
                "action": "VLE Activity Low: Schedule at least 20 minutes of daily course portal reading."
            })
            
        if row['avg_submission_delay'] > 2:
            interventions.append({
                "type": "Time Management",
                "severity": "Medium",
                "action": "Assignment Submission Delay: Set automated calendar alerts 48 hours before deadlines."
            })
            
        if row['avg_assessment_score'] < 55:
            interventions.append({
                "type": "Academic Support",
                "severity": "High",
                "action": "Assessment Score Below Benchmark: Assign interactive diagnostic revision modules."
            })
            
        if row['early_clicks'] < 30:
            interventions.append({
                "type": "Early Warning",
                "severity": "Low",
                "action": "Low Early Course Engagement: Review introductory syllabus video lectures."
            })
            
        if not interventions and prediction == 1:
            interventions.append({
                "type": "Enrichment",
                "severity": "Info",
                "action": "Optimal learning path detected. Recommend advanced extension coursework."
            })

        return {
            "predicted_status": "Pass / Distinction" if prediction == 1 else "At-Risk / Fail",
            "pass_probability": round(prob_pass * 100, 2),
            "risk_score": round((1.0 - prob_pass) * 100, 2),
            "key_influencers": top_factors,
            "personalized_interventions": interventions
        }

if __name__ == "__main__":
    engine = AdaptiveXAIEngine()
    
    # Test sample record
    sample_student = pd.DataFrame([{
        'gender': 'M', 'highest_education': 'HE Qualification', 'imd_band': '90-100%',
        'age_band': '0-35', 'disability': 'N', 'num_of_prev_attempts': 0,
        'studied_credits': 60, 'module_presentation_length': 268, 'date_registration': -50.0,
        'avg_assessment_score': 42.0, 'total_weighted_score': 35.0, 'assessments_completed': 2.0,
        'avg_submission_delay': 4.5, 'total_clicks': 320.0, 'mean_daily_clicks': 2.1,
        'active_vle_days': 12.0, 'early_clicks': 15.0
    }])
    
    explanation = engine.explain_student(sample_student)
    print("XAI Output Result:\n", explanation)