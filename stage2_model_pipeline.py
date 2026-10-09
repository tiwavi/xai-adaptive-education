import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import joblib

def build_data_pipeline(filepath):
    print("[1/5] Loading cleaned dataset...")
    df = pd.read_csv(filepath)
    
    print("[2/5] Engineering domain-specific adaptive learning metrics...")
    # Custom engineered educational analytics
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
    
    return X, y, preprocessor, features, categorical_cols, numerical_cols

def run_stage2_training():
    filepath = 'cleaned_ou_data.csv'
    X, y, preprocessor, feature_names, cat_cols, num_cols = build_data_pipeline(filepath)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[3/5] Preprocessing features and applying 5-Fold Cross Validation...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Production Model: Gradient Boosting
    model = HistGradientBoostingClassifier(
        random_state=42, 
        max_iter=180, 
        learning_rate=0.07, 
        max_leaf_nodes=31
    )
    
    # 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring='accuracy')
    print(f"   -> 5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    
    print("[4/5] Training final production models...")
    model.fit(X_train_proc, y_train)
    
    # Explainer Surrogate Forest (for XAI feature attributions)
    explainer_model = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
    explainer_model.fit(X_train_proc, y_train)
    
    # Evaluation
    preds = model.predict(X_test_proc)
    probs = model.predict_proba(X_test_proc)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print("\n" + "="*50)
    print(" STAGE 2 EVALUATION METRICS")
    print("="*50)
    print(f" Final Test Accuracy : {acc * 100:.2f}%")
    print(f" Final Test AUC-ROC  : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f" True Negatives  (At-Risk correctly predicted) : {cm[0][0]}")
    print(f" False Positives (At-Risk predicted as Pass)  : {cm[0][1]}")
    print(f" False Negatives (Pass predicted as At-Risk)  : {cm[1][0]}")
    print(f" True Positives  (Pass correctly predicted)    : {cm[1][1]}")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("="*50 + "\n")
    
    print("[5/5] Saving production model artifacts...")
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_transformed_features = list(num_cols) + list(cat_feature_names)
    
    joblib.dump(model, 'model.pkl')
    joblib.dump(explainer_model, 'explainer_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(all_transformed_features, 'transformed_feature_names.pkl')
    joblib.dump(feature_names, 'raw_feature_names.pkl')
    
    print(" Stage 2 Complete! Saved: model.pkl, explainer_model.pkl, preprocessor.pkl")

if __name__ == "__main__":
    run_stage2_training()