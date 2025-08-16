# src/trainers/train_rf.py
"""
Train a Random Forest pipeline on the weekly data, evaluate and save the trained model.
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from src.data.load_data import load_weekly
from src.features.preprocessor import get_feature_pipeline
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectFromModel



def main(
    data_dir: str = 'data/raw/anonymisedData',
    artifact_path: str = 'artifacts/rf_pipeline.pkl',
    test_size: float = 0.2,
    random_state: int = 42
):
    # 1. load data
    df = load_weekly(data_dir)
    y = df['target_fail']
    drop_cols = [
        'id_student', 'code_module', 'code_presentation',
        'week_end', 'target_fail'
    ]
    X = df.drop(columns=drop_cols)

    # 2. define features category
    num_feats = [
        *[c for c in X.columns if c.startswith('cum_click_') or c.startswith('sum_click_')],
        'week', 'week_start_day', 'num_of_prev_attempts', 'studied_credits',
        'course_priority_rank', 'course_progress_ratio', 'weeks_to_course_end', 
        'repeat_risk_score', 'study_load_intensity',
        'is_early_stage', 'is_mid_stage', 'is_final_stage'
    ]
    ord_feats = [c for c in X.columns if c in ('highest_education', 'imd_band', 'age_band')]
    nom_feats = [c for c in X.columns if c in ('gender', 'region', 'disability',
                                               'age_education_interaction', 'region_deprivation_interaction')
    ]


    # categorization check
    all_categorized_features = set(num_feats + ord_feats + nom_feats)
    all_features = set(X.columns)
    uncategorized_features = all_features - all_categorized_features

    if uncategorized_features:
            print(f"warning, following features is not categorised: {uncategorized_features}")
        
    print(f"num features ({len(num_feats)}): {num_feats}")
    print(f"ordinal features ({len(ord_feats)}): {ord_feats}")
    print(f"nominal features ({len(nom_feats)}): {nom_feats}")


    # 3. construct prepreocessing and model Pipeline
    preprocessor = get_feature_pipeline(
        num_feats=num_feats,
        ord_feats=ord_feats,
        nom_feats=nom_feats,
        model_type='tree'
    )


    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('smote_tomek', SMOTETomek(random_state=random_state)),  # SMOTE + Tomek Links 
        ('sel', SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=random_state
            ),
            threshold='median'
        )),        
        ('clf', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=random_state
        ))
    ])

# Fast training, cite if full needed
    if len(X) > 100000:  
        sample_size = min(100000, len(X))
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[sample_idx]
        y = y.iloc[sample_idx]
        print(f"Sampled {sample_size} from {len(X)} records for faster training")



    # 4. dataset splitting and fitting model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=test_size,
        random_state=random_state
    )
    pipeline.fit(X_train, y_train)

    # 5. model evaluation
    # predict_proba: fail probability of each input
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 5. evaluation with threshold 0.4
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    thresh = 0.4
    y_pred = (y_proba >= thresh).astype(int)

    print(f"=== threshold = {thresh} ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # 5.1 calculate Log Loss
    from sklearn.metrics import log_loss
    y_proba_matrix = pipeline.predict_proba(X_test)
    ll = log_loss(y_test, y_proba_matrix)
    print(f"Log Loss: {ll:.4f}")

    # 5.2 plot AUC-PR (Precision-Recall Curve)
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap_score:.4f})')
    
    # Add baseline (random classifier) - for imbalanced data, this is the positive class ratio
    baseline = y_test.sum() / len(y_test)
    plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Random Classifier (baseline = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # save PR curve
    pr_plot_path = artifact_path.replace('.pkl', '_pr_curve.png')
    plt.tight_layout()
    plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
    print(f"PR curve saved to {pr_plot_path}")
    plt.show()




    # 6. save model
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    print(f"Model pipeline saved to {artifact_path}")


if __name__ == '__main__':
    main()