import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

def load_and_preprocess_data(filepath):
    print("[1/4] Loading dataset...")
    df = pd.read_csv(filepath)
    
    print("[2/4] Engineering custom educational features...")
    df['clicks_per_active_day'] = df['total_clicks'] / (df['active_vle_days'] + 1e-5)
    df['assessment_completion_ratio'] = df['assessments_completed'] / 14.0
    df['early_engagement_ratio'] = df['early_clicks'] / (df['total_clicks'] + 1.0)
    
    features = [
        'gender', 'highest_education', 'imd_band', 'age_band', 'disability',
        'num_of_prev_attempts', 'studied_credits', 'module_presentation_length',
        'date_registration', 'avg_assessment_score', 'total_weighted_score',
        'assessments_completed', 'avg_submission_delay', 'total_clicks',
        'mean_daily_clicks', 'active_vle_days', 'early_clicks',
        'clicks_per_active_day', 'assessment_completion_ratio', 'early_engagement_ratio'
    ]
    
    X = df[features]
    y = df['target_binary']
    
    categorical_cols = ['gender', 'highest_education', 'imd_band', 'age_band', 'disability']
    numerical_cols = [c for c in features if c not in categorical_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), preprocessor, features, categorical_cols, numerical_cols

def main():
    filepath = 'cleaned_ou_data.csv'
    (X_train, X_test, y_train, y_test), preprocessor, feature_names, cat_cols, num_cols = load_and_preprocess_data(filepath)
    
    print("[3/4] Transforming feature pipelines & training model...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Primary Predictor
    model = HistGradientBoostingClassifier(random_state=42, max_iter=150, learning_rate=0.08)
    model.fit(X_train_proc, y_train)
    
    # Forest Explainer Engine
    explainer_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    explainer_model.fit(X_train_proc, y_train)
    
    print("[4/4] Evaluating model performance...")
    preds = model.predict(X_test_proc)
    probs = model.predict_proba(X_test_proc)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    print("\n" + "="*45)
    print(f" Model Accuracy : {acc * 100:.2f}%")
    print(f" Model AUC-ROC  : {auc:.4f}")
    print("="*45 + "\n")
    print("Classification Report:\n", classification_report(y_test, preds))
    
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_transformed_features = list(num_cols) + list(cat_feature_names)
    
    # Save artifacts
    joblib.dump(model, 'model.pkl')
    joblib.dump(explainer_model, 'explainer_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(all_transformed_features, 'transformed_feature_names.pkl')
    joblib.dump(feature_names, 'raw_feature_names.pkl')
    print(" Artifacts successfully saved: model.pkl, explainer_model.pkl, preprocessor.pkl")

if __name__ == "__main__":
    main()