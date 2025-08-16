# src/trainers/train_logistic_final.py
"""
Final optimized Logistic Regression based on lessons learned from previous attempts.
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from src.data.load_data import load_weekly
from src.features.preprocessor import get_feature_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import seaborn as sns
import gc
import json
import time
from datetime import datetime


def create_balanced_features(df):
    """
    Create a balanced set of features based on domain knowledge and previous experiments
    """
    df_new = df.copy()
    
    # 1. Key polynomial features (only the most important ones)
    df_new['course_progress_squared'] = df_new['course_progress_ratio'] ** 2
    df_new['repeat_risk_squared'] = df_new['repeat_risk_score'] ** 2
    
    # 2. Critical interaction features
    df_new['progress_risk_interaction'] = (
        df_new['course_progress_ratio'] * df_new['repeat_risk_score']
    )
    df_new['progress_weeks_interaction'] = (
        df_new['course_progress_ratio'] * df_new['weeks_to_course_end']
    )
    
    # 3. Engagement features (log transform key click features)
    key_click_features = ['sum_click_quiz', 'sum_click_oucontent', 'sum_click_forumng']
    for col in key_click_features:
        if col in df_new.columns:
            df_new[f'{col}_log1p'] = np.log1p(df_new[col])
    
    # 4. Engagement ratios
    df_new['quiz_content_ratio'] = (
        df_new['sum_click_quiz'] / (df_new['sum_click_oucontent'] + 1)
    )
    df_new['active_passive_ratio'] = (
        (df_new['sum_click_quiz'] + df_new['sum_click_forumng']) / 
        (df_new['sum_click_resource'] + df_new['sum_click_page'] + 1)
    )
    
    # 5. Overall engagement score
    engagement_cols = ['sum_click_quiz', 'sum_click_oucontent', 'sum_click_forumng']
    df_new['total_engagement'] = df_new[engagement_cols].sum(axis=1)
    df_new['total_engagement_log1p'] = np.log1p(df_new['total_engagement'])
    
    # 6. Risk indicators (binary features)
    df_new['high_repeat_risk'] = (df_new['repeat_risk_score'] > df_new['repeat_risk_score'].quantile(0.75)).astype(int)
    df_new['low_progress'] = (df_new['course_progress_ratio'] < 0.4).astype(int)
    df_new['minimal_engagement'] = (df_new['total_engagement'] < df_new['total_engagement'].quantile(0.25)).astype(int)
    df_new['late_in_course'] = (df_new['weeks_to_course_end'] < 5).astype(int)
    
    return df_new


def evaluate_at_multiple_thresholds(y_test, y_proba):
    """
    Comprehensive threshold evaluation focusing on practical usability
    """
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        if len(np.unique(y_pred)) < 2:  # If all predictions are the same class
            continue
            
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Business metrics
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'false_alarm_rate': false_alarm_rate,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision_recall_balance': abs(precision - recall),
            'business_value': precision * recall * (1 - false_alarm_rate)  # Custom metric
        })
    
    return pd.DataFrame(results)


def find_best_threshold_strategies(threshold_df):
    """
    Find thresholds optimized for different business scenarios
    """
    strategies = {}
    
    # Strategy 1: Best F1-score
    if not threshold_df.empty and threshold_df['f1_score'].max() > 0:
        best_f1_idx = threshold_df['f1_score'].idxmax()
        strategies['best_f1'] = {
            'threshold': threshold_df.loc[best_f1_idx, 'threshold'],
            'metrics': threshold_df.loc[best_f1_idx].to_dict()
        }
    
    # Strategy 2: Best precision-recall balance
    if not threshold_df.empty:
        balanced_idx = threshold_df['precision_recall_balance'].idxmin()
        strategies['balanced'] = {
            'threshold': threshold_df.loc[balanced_idx, 'threshold'],
            'metrics': threshold_df.loc[balanced_idx].to_dict()
        }
    
    # Strategy 3: High precision (at least 25%)
    high_precision_df = threshold_df[threshold_df['precision'] >= 0.25]
    if not high_precision_df.empty:
        # Among high precision options, choose the one with best recall
        best_recall_idx = high_precision_df['recall'].idxmax()
        strategies['high_precision'] = {
            'threshold': high_precision_df.loc[best_recall_idx, 'threshold'],
            'metrics': high_precision_df.loc[best_recall_idx].to_dict()
        }
    
    # Strategy 4: Business value optimization
    if not threshold_df.empty and threshold_df['business_value'].max() > 0:
        best_business_idx = threshold_df['business_value'].idxmax()
        strategies['business_optimized'] = {
            'threshold': threshold_df.loc[best_business_idx, 'threshold'],
            'metrics': threshold_df.loc[best_business_idx].to_dict()
        }
    
    return strategies


def plot_comprehensive_analysis(threshold_df, strategies, save_path=None):
    """
    Comprehensive visualization of threshold analysis
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Main metrics
    ax1.plot(threshold_df['threshold'], threshold_df['precision'], 'b-', label='Precision', marker='o', alpha=0.7)
    ax1.plot(threshold_df['threshold'], threshold_df['recall'], 'r-', label='Recall', marker='s', alpha=0.7)
    ax1.plot(threshold_df['threshold'], threshold_df['f1_score'], 'g-', label='F1-Score', marker='^', alpha=0.7)
    
    # Mark strategy points
    for strategy_name, strategy_info in strategies.items():
        thresh = strategy_info['threshold']
        f1 = strategy_info['metrics']['f1_score']
        ax1.axvline(x=thresh, color='purple', linestyle='--', alpha=0.5)
        ax1.text(thresh, f1 + 0.02, strategy_name, rotation=90, fontsize=8)
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, and F1-Score vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Business metrics
    ax2.plot(threshold_df['threshold'], threshold_df['false_alarm_rate'], 'orange', label='False Alarm Rate', marker='d')
    ax2.plot(threshold_df['threshold'], threshold_df['business_value'], 'purple', label='Business Value', marker='*')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Rate/Value')
    ax2.set_title('Business Metrics vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion matrix components (log scale)
    ax3.semilogy(threshold_df['threshold'], threshold_df['tp'], 'g-', label='True Positives', marker='o')
    ax3.semilogy(threshold_df['threshold'], threshold_df['fp'], 'r-', label='False Positives', marker='s')
    ax3.semilogy(threshold_df['threshold'], threshold_df['fn'], 'orange', label='False Negatives', marker='^')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Count (Log Scale)')
    ax3.set_title('Confusion Matrix Components')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Precision-Recall curve style
    ax4.plot(threshold_df['recall'], threshold_df['precision'], 'b-', marker='o', alpha=0.7)
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision vs Recall')
    ax4.grid(True, alpha=0.3)
    
    # Mark strategy points on PR curve
    for strategy_name, strategy_info in strategies.items():
        recall = strategy_info['metrics']['recall']
        precision = strategy_info['metrics']['precision']
        ax4.plot(recall, precision, 'ro', markersize=8, alpha=0.7)
        ax4.text(recall + 0.01, precision + 0.01, strategy_name, fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive analysis plot saved to {save_path}")
    plt.show()


def extract_feature_importance(pipeline, model_name="logistic_regression"):
    """
    Extract feature importance from the trained pipeline
    """
    try:
        # Get feature names after preprocessing
        preprocessor = pipeline.named_steps['preproc']
        
        # Get selected features
        if 'feature_select' in pipeline.named_steps:
            feature_selector = pipeline.named_steps['feature_select']
            selected_features_mask = feature_selector.get_support()
        else:
            selected_features_mask = None
        
        # Get feature names after preprocessing
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            # Fallback if get_feature_names_out is not available
            feature_names = [f"feature_{i}" for i in range(preprocessor.transform(pipeline.named_steps['preproc'].fit_transform(X_train[:1])).shape[1])]
        
        # Apply feature selection mask if exists
        if selected_features_mask is not None:
            feature_names = [name for name, selected in zip(feature_names, selected_features_mask) if selected]
        
        # Get coefficients from logistic regression
        classifier = pipeline.named_steps['clf']
        if hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])  # Use absolute values for importance
        else:
            importances = np.zeros(len(feature_names))
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'abs_importance': np.abs(importances)
        }).sort_values('abs_importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        print(f"Warning: Could not extract feature importance: {e}")
        # Return empty dataframe with expected structure
        return pd.DataFrame({
            'feature': ['unknown'],
            'importance': [0.0],
            'abs_importance': [0.0]
        })


def save_model_results(
    model_name: str,
    pipeline,
    X_test, y_test, 
    y_proba,
    threshold_results,
    strategies,
    cv_scores,
    training_time,
    config_dict,
    test_student_ids=None
):
    """
    Save comprehensive model results in standardized format
    """
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    
    base_path = f"artifacts/{model_name}"
    
    # 1. Performance metrics
    best_f1_metrics = strategies.get('best_f1', {}).get('metrics', {})
    balanced_metrics = strategies.get('balanced', {}).get('metrics', {})
    
    metrics = {
        "model_name": model_name,
        "test_auc": float(roc_auc_score(y_test, y_proba)),
        "test_log_loss": float(log_loss(y_test, np.column_stack([1-y_proba, y_proba]))),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "best_f1_score": float(best_f1_metrics.get('f1_score', 0)),
        "best_f1_threshold": float(best_f1_metrics.get('threshold', 0.5)),
        "best_f1_precision": float(best_f1_metrics.get('precision', 0)),
        "best_f1_recall": float(best_f1_metrics.get('recall', 0)),
        "balanced_threshold": float(balanced_metrics.get('threshold', 0.5)),
        "balanced_f1": float(balanced_metrics.get('f1_score', 0)),
        "training_time_seconds": float(training_time),
        "n_test_samples": int(len(y_test)),
        "n_training_samples": int(len(y_test) * 4),  # Assuming test is 20%
        "n_total_features": int(config_dict.get('n_total_features', 0)),
        "timestamp": datetime.now().isoformat(),
        "config": config_dict
    }
    
    # Try to get number of selected features
    try:
        if 'feature_select' in pipeline.named_steps:
            metrics["n_selected_features"] = int(pipeline.named_steps['feature_select'].get_support().sum())
        else:
            metrics["n_selected_features"] = metrics["n_total_features"]
    except:
        metrics["n_selected_features"] = 0
    
    with open(f"{base_path}_performance_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 2. Predictions with multiple thresholds
    predictions_data = {
        'true_label': y_test.astype(int),
        'predicted_proba': y_proba.astype(float)
    }
    
    # Add predictions for different strategies
    for strategy_name, strategy_info in strategies.items():
        threshold = strategy_info['threshold']
        pred_col_name = f'predicted_{strategy_name}'
        predictions_data[pred_col_name] = (y_proba >= threshold).astype(int)
    
    # Add student IDs if available
    if test_student_ids is not None:
        predictions_data['student_id'] = test_student_ids
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv(f"{base_path}_predictions.csv", index=False)
    
    # 3. Feature importance
    importance_df = extract_feature_importance(pipeline, model_name)
    importance_df.to_csv(f"{base_path}_feature_importance.csv", index=False)
    
    # 4. Configuration file
    with open(f"{base_path}_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 5. Threshold analysis (already exists, just ensure it's saved)
    threshold_results.to_csv(f"{base_path}_threshold_analysis.csv", index=False)
    
    # 6. Strategies summary
    strategies_clean = {}
    for strategy_name, strategy_info in strategies.items():
        strategies_clean[strategy_name] = {
            'threshold': float(strategy_info['threshold']),
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in strategy_info['metrics'].items()}
        }
    
    with open(f"{base_path}_strategies.json", 'w') as f:
        json.dump(strategies_clean, f, indent=2)
    
    # Also save as text for readability
    with open(f"{base_path}_strategies.txt", 'w') as f:
        for strategy_name, strategy_info in strategies.items():
            f.write(f"{strategy_name.upper()}:\n")
            f.write(f"Threshold: {strategy_info['threshold']:.3f}\n")
            f.write(f"Metrics: {strategy_info['metrics']}\n\n")
    
    print(f"All results saved with base path: {base_path}")
    return base_path


def main(
    data_dir: str = 'data/raw/anonymisedData',
    artifact_path: str = 'artifacts/logistic_final_pipeline.pkl',
    test_size: float = 0.2,
    random_state: int = 42,
    sampling_strategy: str = 'moderate_smote',  # 'none', 'moderate_smote', 'undersampling'
    max_samples: int = None
):
    # Record start time
    start_time = time.time()
    
    # 1. Load and prepare data
    print("Loading data...")
    df = load_weekly(data_dir)
    
    # Optimize data types
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    print(f"Original data shape: {df.shape}")
    
    # Sample if needed
    if max_samples and len(df) > max_samples:
        df_sampled, _ = train_test_split(
            df, train_size=max_samples, stratify=df['target_fail'], random_state=random_state
        )
        df = df_sampled
        print(f"Sampled to {max_samples} rows")
    
    # Create features
    print("Creating balanced feature set...")
    df = create_balanced_features(df)
    
    y = df['target_fail']
    drop_cols = ['id_student', 'code_module', 'code_presentation', 'week_end', 'target_fail']
    
    # Store student IDs for predictions (before dropping)
    student_ids = df['id_student'].values if 'id_student' in df.columns else None
    
    X = df.drop(columns=drop_cols)
    
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # 2. Define features
    num_feats = [
        # Core numerical features
        *[c for c in X.columns if c.startswith('cum_click_') or c.startswith('sum_click_')],
        'week', 'week_start_day', 'num_of_prev_attempts', 'studied_credits',
        'course_priority_rank', 'course_progress_ratio', 'weeks_to_course_end', 
        'repeat_risk_score', 'study_load_intensity',
        'is_early_stage', 'is_mid_stage', 'is_final_stage',
        # New features
        'course_progress_squared', 'repeat_risk_squared',
        'progress_risk_interaction', 'progress_weeks_interaction',
        'quiz_content_ratio', 'active_passive_ratio',
        'total_engagement', 'total_engagement_log1p',
        'high_repeat_risk', 'low_progress', 'minimal_engagement', 'late_in_course'
    ]
    
    # Add log features
    log_features = [c for c in X.columns if c.endswith('_log1p')]
    num_feats.extend(log_features)
    
    ord_feats = [c for c in X.columns if c in ('highest_education', 'imd_band', 'age_band')]
    nom_feats = [c for c in X.columns if c in (
        'gender', 'region', 'disability', 'age_education_interaction', 'region_deprivation_interaction'
    )]
    
    # Handle uncategorized features
    all_categorized = set(num_feats + ord_feats + nom_feats)
    all_features = set(X.columns)
    uncategorized = all_features - all_categorized
    if uncategorized:
        print(f"Adding uncategorized features to numerical: {uncategorized}")
        num_feats.extend(list(uncategorized))
    
    print(f"Total features: {len(num_feats + ord_feats + nom_feats)}")

    # 3. Create preprocessing pipeline
    preprocessor = get_feature_pipeline(
        num_feats=num_feats, ord_feats=ord_feats, nom_feats=nom_feats, model_type='logistic'
    )

    # 4. Choose sampling strategy
    if sampling_strategy == 'none':
        print("Using no sampling, only class balancing")
        sampler = None
    elif sampling_strategy == 'moderate_smote':
        print("Using moderate SMOTE")
        sampler = SMOTE(sampling_strategy=0.4, random_state=random_state, k_neighbors=5)
    elif sampling_strategy == 'undersampling':
        print("Using undersampling")
        sampler = RandomUnderSampler(sampling_strategy=0.6, random_state=random_state)
    else:
        sampler = None

    # 5. Create classifier with moderate regularization
    classifier = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,  # Balanced L1/L2
        C=0.5,  # Moderate regularization (not too strong)
        class_weight='balanced',
        random_state=random_state,
        max_iter=1000,
        n_jobs=-1
    )

    # 6. Create pipeline
    if sampler is not None:
        pipeline = ImbPipeline([
            ('preproc', preprocessor),
            ('sampling', sampler),
            ('feature_select', SelectFromModel(
                LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=random_state),
                threshold='0.5*median'  # Moderate feature selection
            )),
            ('clf', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('preproc', preprocessor),
            ('feature_select', SelectFromModel(
                LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=random_state),
                threshold='0.5*median'
            )),
            ('clf', classifier)
        ])

    # Prepare configuration dictionary
    config_dict = {
        "model_type": "logistic_regression",
        "sampling_strategy": sampling_strategy,
        "test_size": test_size,
        "random_state": random_state,
        "max_samples": max_samples,
        "n_total_features": len(num_feats + ord_feats + nom_feats),
        "n_numerical_features": len(num_feats),
        "n_ordinal_features": len(ord_feats),
        "n_nominal_features": len(nom_feats),
        "classifier_params": classifier.get_params(),
        "preprocessing_pipeline": str(preprocessor),
        "feature_selection": "SelectFromModel with L1 regularization"
    }

    # 7. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    
    # Split student IDs accordingly if available
    if student_ids is not None:
        _, test_student_ids = train_test_split(
            student_ids, test_size=test_size, stratify=y, random_state=random_state
        )
    else:
        test_student_ids = None
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Prepare configuration dictionary
    config_dict = {
        "model_type": "logistic_regression",
        "sampling_strategy": sampling_strategy,
        "test_size": test_size,
        "random_state": random_state,
        "max_samples": max_samples,
        "n_total_features": len(num_feats + ord_feats + nom_feats),
        "n_numerical_features": len(num_feats),
        "n_ordinal_features": len(ord_feats),
        "n_nominal_features": len(nom_feats),
        "classifier_params": classifier.get_params(),
        "preprocessing_pipeline": str(preprocessor),
        "feature_selection": "SelectFromModel with L1 regularization"
    }

    # Clear memory
    del df, X
    gc.collect()

    # 8. Train model
    print(f"Training Logistic Regression with {sampling_strategy}...")
    train_start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - train_start_time

    # 9. Evaluation
    print("Evaluating model...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    test_auc = roc_auc_score(y_test, y_proba)
    test_log_loss = log_loss(y_test, pipeline.predict_proba(X_test))
    print(f"ROC AUC: {test_auc:.4f}")
    print(f"Log Loss: {test_log_loss:.4f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Comprehensive threshold analysis
    threshold_results = evaluate_at_multiple_thresholds(y_test, y_proba)
    
    if threshold_results.empty:
        print("ERROR: No valid threshold results found!")
        return
    
    print("\n=== Threshold Analysis Results ===")
    print(threshold_results[['threshold', 'precision', 'recall', 'f1_score', 'tp', 'fp']].round(3))
    
    # Find optimal strategies
    strategies = find_best_threshold_strategies(threshold_results)
    
    print(f"\n=== Recommended Threshold Strategies ===")
    for strategy_name, strategy_info in strategies.items():
        thresh = strategy_info['threshold']
        metrics = strategy_info['metrics']
        print(f"\n{strategy_name.upper()}:")
        print(f"  Threshold: {thresh:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  True Positives: {metrics['tp']}")

    # 10. Save comprehensive results
    print("\n=== Saving Comprehensive Results ===")
    model_name = "logistic_regression"
    base_path = save_model_results(
        model_name=model_name,
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        y_proba=y_proba,
        threshold_results=threshold_results,
        strategies=strategies,
        cv_scores=cv_scores,
        training_time=training_time,
        config_dict=config_dict,
        test_student_ids=test_student_ids
    )

    # 11. Visualizations (keep existing plots)
    artifact_base_path = artifact_path.replace('.pkl', '')
    
    # Comprehensive analysis plot
    analysis_plot_path = f"{artifact_base_path}_comprehensive_analysis.png"
    plot_comprehensive_analysis(threshold_results, strategies, analysis_plot_path)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Final Logistic Regression')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_plot_path = f"{artifact_base_path}_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    # 12. Save pipeline (keep existing)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    print(f"Pipeline saved to {artifact_path}")


    
if __name__ == '__main__':
    # Test the most promising approach
    main(
        sampling_strategy='moderate_smote',  # Try moderate SMOTE first
        max_samples=200000
    )