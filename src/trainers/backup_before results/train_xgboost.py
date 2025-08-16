# src/trainers/train_xgboost.py
"""
Train an XGBoost pipeline with Scale Position Weight on the weekly data, 
evaluate and save the trained model.
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, log_loss
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from imblearn.pipeline import Pipeline
from src.data.load_data import load_weekly
from src.features.preprocessor import get_feature_pipeline


def main(
    data_dir: str = 'data/raw/anonymisedData',
    artifact_path: str = 'artifacts/xgb_pipeline.pkl',
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

    # 3. calculate class imbalance ratio for scale_pos_weight
    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Class imbalance ratio (negative/positive): {pos_weight:.2f}")
    print(f"Using scale_pos_weight: {pos_weight:.2f}")

    # 4. construct preprocessing and XGBoost Pipeline
    preprocessor = get_feature_pipeline(
        num_feats=num_feats,
        ord_feats=ord_feats,
        nom_feats=nom_feats,
        model_type='xgboost'  # Use XGBoost-specific preprocessing
    )

    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('sel', SelectFromModel(
            estimator=xgb.XGBClassifier(
                n_estimators=100,
                scale_pos_weight=pos_weight,
                random_state=random_state,
                eval_metric='logloss'
            ),
            threshold='median'
        )),        
        ('clf', xgb.XGBClassifier(
            n_estimators=300,      # Increased from 200
            max_depth=7,           # Increased from 6  
            learning_rate=0.08,    # Decreased from 0.1 for better convergence
            subsample=0.85,        # Increased from 0.8
            colsample_bytree=0.85, # Increased from 0.8
            scale_pos_weight=pos_weight,  # Handle class imbalance
            reg_alpha=0.05,        # Reduced L1 regularization
            reg_lambda=0.8,        # Reduced L2 regularization  
            min_child_weight=3,    # Added: minimum sum of instance weight in a child
            gamma=0.1,             # Added: minimum loss reduction for split
            random_state=random_state,
            eval_metric='logloss',
            n_jobs=-1
        ))
    ])

    # 5. dataset splitting and fitting model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"Training set size: {len(X_train):,}")
    print(f"Test set size: {len(X_test):,}")
    
    # Fit the pipeline
    print("Training XGBoost model...")
    pipeline.fit(X_train, y_train)

    # 6. model evaluation
    # predict_proba: fail probability of each input
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 6.1 calculate basic metrics
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # 6.2 calculate Log Loss
    y_proba_matrix = pipeline.predict_proba(X_test)
    ll = log_loss(y_test, y_proba_matrix)
    print(f"Log Loss: {ll:.4f}")

    # 6.3 find optimal threshold based on F1 score
    from sklearn.metrics import precision_recall_curve, f1_score
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    
    print(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")

    # 6.4 evaluation with multiple thresholds
    test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, best_threshold]
    
    print(f"\n=== Multiple Threshold Evaluation ===")
    results_summary = []
    
    for thresh in test_thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision_val = precision_score(y_test, y_pred)
        recall_val = recall_score(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred)
        
        results_summary.append({
            'threshold': thresh,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1_val
        })
        
        print(f"\nThreshold = {thresh:.3f}:")
        print(f"  Precision: {precision_val:.4f}")
        print(f"  Recall: {recall_val:.4f}")
        print(f"  F1-Score: {f1_val:.4f}")
        
        if thresh == 0.4:
            print("  --- Detailed Classification Report ---")
            print(classification_report(y_test, y_pred))
    
    # Create threshold comparison table
    results_df = pd.DataFrame(results_summary)
    print(f"\n=== Threshold Comparison Summary ===")
    print(results_df.round(4).to_string(index=False))

    # 6.5 plot evaluation curves with threshold analysis
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('XGBoost - ROC Curve')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve  
    axes[0, 1].plot(recall, precision, color='darkgreen', lw=2, label='PR curve')
    axes[0, 1].axvline(x=recall[best_threshold_idx], color='red', linestyle='--', 
                      label=f'Best threshold: {best_threshold:.3f}')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('XGBoost - Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Threshold vs F1 Score
    axes[1, 0].plot(thresholds, f1_scores[:-1], color='purple', lw=2)
    axes[1, 0].axvline(x=best_threshold, color='red', linestyle='--', 
                      label=f'Best F1: {best_f1:.4f}')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Threshold vs F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Threshold Comparison Bar Chart
    thresh_comparison = results_df[results_df['threshold'].isin([0.3, 0.4, 0.5, 0.6, 0.7])]
    x_pos = np.arange(len(thresh_comparison))
    width = 0.25
    
    axes[1, 1].bar(x_pos - width, thresh_comparison['precision'], width, label='Precision', alpha=0.8)
    axes[1, 1].bar(x_pos, thresh_comparison['recall'], width, label='Recall', alpha=0.8)
    axes[1, 1].bar(x_pos + width, thresh_comparison['f1_score'], width, label='F1-Score', alpha=0.8)
    
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance by Threshold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'{t:.1f}' for t in thresh_comparison['threshold']])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save plots
    roc_plot_path = artifact_path.replace('.pkl', '_evaluation_curves.png')
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation curves saved to {roc_plot_path}")
    plt.show()

    # 7. feature importance analysis
    print("\n=== Feature Importance Analysis ===")
    
    # Get feature names after preprocessing
    preprocessor_fitted = pipeline.named_steps['preproc']
    feature_names = []
    
    # Add numerical features
    feature_names.extend(num_feats)
    
    # Add ordinal features  
    feature_names.extend(ord_feats)
    
    # Add nominal features (one-hot encoded)
    if hasattr(preprocessor_fitted.named_transformers_['nom'].named_steps['onehot'], 'get_feature_names_out'):
        nom_features = preprocessor_fitted.named_transformers_['nom'].named_steps['onehot'].get_feature_names_out(nom_feats)
        feature_names.extend(nom_features)
    
    # Get selected features after feature selection
    selector = pipeline.named_steps['sel']
    selected_features = np.array(feature_names)[selector.get_support()]
    
    # Get feature importance from XGBoost
    xgb_model = pipeline.named_steps['clf']
    importances = xgb_model.feature_importances_
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost - Top 20 Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save feature importance plot
    importance_plot_path = artifact_path.replace('.pkl', '_feature_importance.png')
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {importance_plot_path}")
    plt.show()

    # 8. save model
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    print(f"XGBoost pipeline saved to {artifact_path}")

    # 9. save feature importance
    importance_path = artifact_path.replace('.pkl', '_feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")

    # 10. model summary with recommendations
    print(f"\n=== XGBoost Model Summary ===")
    print(f"Scale Position Weight: {pos_weight:.2f}")
    print(f"Number of features after selection: {len(selected_features)}")
    print(f"ROC AUC: {auc_score:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print(f"Best F1 threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    
    # Performance recommendations
    print(f"\n=== Performance Analysis & Recommendations ===")
    
    # Find best performing thresholds for different metrics
    best_precision_idx = results_df['precision'].idxmax()
    best_recall_idx = results_df['recall'].idxmax()
    best_f1_idx = results_df['f1_score'].idxmax()
    
    print(f"• Best Precision: {results_df.loc[best_precision_idx, 'precision']:.4f} at threshold {results_df.loc[best_precision_idx, 'threshold']:.3f}")
    print(f"• Best Recall: {results_df.loc[best_recall_idx, 'recall']:.4f} at threshold {results_df.loc[best_recall_idx, 'threshold']:.3f}")
    print(f"• Best F1-Score: {results_df.loc[best_f1_idx, 'f1_score']:.4f} at threshold {results_df.loc[best_f1_idx, 'threshold']:.3f}")
    
    # Threshold recommendations based on use case
    print(f"\n=== Threshold Recommendations by Use Case ===")
    print(f"• Conservative (High Precision): Use threshold ≥ 0.6 (fewer false alarms)")
    print(f"• Balanced: Use threshold around 0.4-0.5 (balance precision/recall)")  
    print(f"• Sensitive (High Recall): Use threshold ≤ 0.3 (catch more at-risk students)")
    print(f"• Optimal F1: Use threshold {best_threshold:.3f}")
    
    # Model performance assessment
    if auc_score >= 0.85:
        performance_level = "Excellent"
    elif auc_score >= 0.80:
        performance_level = "Good"
    elif auc_score >= 0.75:
        performance_level = "Fair"
    else:
        performance_level = "Needs Improvement"
    
    print(f"\n=== Overall Model Assessment ===")
    print(f"Model Performance: {performance_level} (ROC AUC: {auc_score:.4f})")
    
    if best_f1 < 0.4:
        print("⚠️  F1-Score is relatively low - consider:")
        print("   - Collecting more positive samples")
        print("   - Feature engineering improvements") 
        print("   - Hyperparameter tuning")
        print("   - Ensemble methods")
    elif best_f1 < 0.6:
        print("✓ F1-Score is reasonable but can be improved")
    else:
        print("✓ F1-Score shows good model performance")
        
    # Save threshold analysis
    threshold_analysis_path = artifact_path.replace('.pkl', '_threshold_analysis.csv')
    results_df.to_csv(threshold_analysis_path, index=False)
    print(f"\nThreshold analysis saved to {threshold_analysis_path}")


if __name__ == '__main__':
    main()