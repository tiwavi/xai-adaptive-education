import joblib
import pandas as pd
import numpy as np

class AdaptiveXAIEngine:
    """
    Advanced Explainable AI Engine for Personalized & Adaptive Education.
    Computes local feature importances, generates learning pathways, 
    and simulates 'What-If' scenarios.
    """
    def __init__(self, model_path='model.pkl', explainer_path='explainer_model.pkl', 
                 preprocessor_path='preprocessor.pkl', features_path='transformed_feature_names.pkl'):
        self.model = joblib.load(model_path)
        self.explainer_model = joblib.load(explainer_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_names = joblib.load(features_path)
        
    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['clicks_per_active_day'] = df['total_clicks'] / (df['active_vle_days'] + 1e-5)
        df['assessment_completion_ratio'] = df['assessments_completed'] / 14.0
        df['early_engagement_ratio'] = df['early_clicks'] / (df['total_clicks'] + 1.0)
        return df

    def explain_student(self, student_df: pd.DataFrame) -> dict:
        processed_df = self._preprocess_df(student_df)
        X_proc = self.preprocessor.transform(processed_df)
        
        prob_pass = float(self.model.predict_proba(X_proc)[0][1])
        prediction = int(self.model.predict(X_proc)[0])
        
        importances = self.explainer_model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:5]
        
        top_factors = []
        for idx in top_indices:
            clean_name = self.feature_names[idx].replace("num__", "").replace("cat__", "")
            top_factors.append({
                "feature": clean_name,
                "importance_percentage": round(float(importances[idx]) * 100, 2)
            })
            
        row = processed_df.iloc[0]
        
        # Determine Adaptive Learning Pathway
        pathway = self._generate_adaptive_pathway(prob_pass, row)
        
        # Interventions
        interventions = self._generate_interventions(row, prediction)

        return {
            "predicted_status": "Pass / Distinction" if prediction == 1 else "At-Risk / Fail",
            "pass_probability": round(prob_pass * 100, 2),
            "risk_score": round((1.0 - prob_pass) * 100, 2),
            "adaptive_pathway": pathway,
            "key_influencers": top_factors,
            "personalized_interventions": interventions
        }

    def simulate_what_if(self, student_df: pd.DataFrame, delta_vle_days: float = 0, delta_score: float = 0) -> dict:
        """Simulate how changing behavior impacts passing probability."""
        modified_df = student_df.copy()
        modified_df['active_vle_days'] = modified_df['active_vle_days'] + delta_vle_days
        modified_df['avg_assessment_score'] = np.clip(modified_df['avg_assessment_score'] + delta_score, 0, 100)
        
        baseline_res = self.explain_student(student_df)
        simulated_res = self.explain_student(modified_df)
        
        prob_diff = round(simulated_res['pass_probability'] - baseline_res['pass_probability'], 2)
        
        return {
            "baseline_probability": baseline_res['pass_probability'],
            "simulated_probability": simulated_res['pass_probability'],
            "probability_gain": prob_diff,
            "simulated_status": simulated_res['predicted_status']
        }

    def _generate_adaptive_pathway(self, prob_pass: float, row) -> dict:
        if prob_pass < 0.40:
            return {
                "track": "Remedial Intensive Acceleration",
                "focus": "Core concept reinforcement & active portal engagement",
                "recommended_modules": ["Foundational Diagnostic Quiz", "Syllabus Re-engagement Unit", "Weekly Guided Peer Tutoring"],
                "target_weekly_vle_hours": 8
            }
        elif prob_pass < 0.70:
            return {
                "track": "Guided Adaptive Support",
                "focus": "Assignment deadline compliance & submission consistency",
                "recommended_modules": ["Mid-term Review Workshops", "Structured Problem Sets", "Time Management Routine"],
                "target_weekly_vle_hours": 5
            }
        else:
            return {
                "track": "Advanced Extension & Leadership",
                "focus": "Deep concept mastery & extension challenge projects",
                "recommended_modules": ["Advanced Case Studies", "Peer Mentorship Program", "Independent Research Track"],
                "target_weekly_vle_hours": 3
            }

    def _generate_interventions(self, row, prediction: int) -> list:
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
        return interventions