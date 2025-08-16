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
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_curve, auc, log_loss, matthews_corrcoef, fbeta_score
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from src.data.load_data import load_weekly
from src.features.preprocessor import get_feature_pipeline
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectFromModel
# === NEW: unified reporting helpers ===
from src.trainers.unified_reporting import (
    export_dataset_summary,
    export_pr_curve,
    threshold_sweep,
    export_feature_importance,
    error_analysis,
    simulate_intervention,
)


def evaluate_with_optimal_threshold(y_test, y_proba, default_thresh=0.4):
    """
    Evaluate model with both default threshold and optimal F1 threshold
    Returns evaluation results and optimal threshold info
    """
    # Find optimal F1 threshold
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]
    
    print(f"\n=== OPTIMAL F1 THRESHOLD FOUND ===")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Optimal F1-score: {optimal_f1:.4f}")
    
    # Evaluate at default threshold
    y_pred_default = (y_proba >= default_thresh).astype(int)
    print(f"\n=== EVALUATION AT DEFAULT THRESHOLD = {default_thresh} ===")
    print(classification_report(y_test, y_pred_default))
    
    # Evaluate at optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    print(f"\n=== EVALUATION AT OPTIMAL THRESHOLD = {optimal_threshold:.4f} ===")
    print(classification_report(y_test, y_pred_optimal))
    
    # Calculate metrics for both thresholds
    def get_metrics(y_true, y_pred, thresh_name):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        return {
            'threshold_name': thresh_name,
            'confusion_matrix': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'mcc': mcc
        }
    
    default_metrics = get_metrics(y_test, y_pred_default, f'default_{default_thresh}')
    optimal_metrics = get_metrics(y_test, y_pred_optimal, f'optimal_{optimal_threshold:.4f}')
    
    return {
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'default_metrics': default_metrics,
        'optimal_metrics': optimal_metrics,
        'thresholds_for_sweep': [0.3, 0.4, 0.5, 0.6, 0.7, optimal_threshold]
    }


def plot_enhanced_curves(y_test, y_proba, artifact_path, optimal_threshold=None):
    """
    Plot enhanced PR and ROC curves with AUC values and optimal threshold marking
    """
    # Calculate metrics
    auroc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    base_rate = y_test.mean()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # PR Curve (Enhanced)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
    ax1.plot(recalls, precisions, lw=2, label=f'PR curve (AUC = {ap:.4f})')
    ax1.axhline(y=base_rate, color='gray', lw=1.5, linestyle='--', 
               label=f'Baseline = {base_rate:.3f}')
    
    # Mark optimal threshold on PR curve if provided
    if optimal_threshold is not None:
        # Find the point on PR curve closest to optimal threshold
        if len(pr_thresholds) > 0:
            # Find closest threshold
            thresh_diffs = np.abs(pr_thresholds - optimal_threshold)
            closest_idx = np.argmin(thresh_diffs)
            if closest_idx < len(precisions) and closest_idx < len(recalls):
                opt_precision = precisions[closest_idx]
                opt_recall = recalls[closest_idx]
                ax1.plot(opt_recall, opt_precision, 'ro', markersize=8, 
                        label=f'Optimal F1 @ {optimal_threshold:.3f}')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(f'Precision-Recall Curve (AUC = {ap:.4f})')
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)
    
    # ROC Curve (Enhanced)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    ax2.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auroc:.4f})')
    ax2.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random')
    
    # Mark optimal threshold on ROC curve if provided
    if optimal_threshold is not None and len(roc_thresholds) > 0:
        # Find closest threshold
        thresh_diffs = np.abs(roc_thresholds - optimal_threshold)
        closest_idx = np.argmin(thresh_diffs)
        if closest_idx < len(fpr) and closest_idx < len(tpr):
            opt_fpr = fpr[closest_idx]
            opt_tpr = tpr[closest_idx]
            ax2.plot(opt_fpr, opt_tpr, 'ro', markersize=8,
                    label=f'Optimal F1 @ {optimal_threshold:.3f}')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'ROC Curve (AUC = {auroc:.4f})')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    # Save combined plot
    plt.tight_layout()
    curves_path = artifact_path.replace('.pkl', '_evaluation_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced evaluation curves saved to {curves_path}")
    print(f"AUROC: {auroc:.4f}, AUC-PR: {ap:.4f}")
    plt.show()
    
    return auroc, ap


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
        print(f"Warning: Following features are not categorized: {uncategorized_features}")
        # Add uncategorized features to numerical features
        num_feats.extend(list(uncategorized_features))
        
    print(f"Numerical features ({len(num_feats)}): {num_feats}")
    print(f"Ordinal features ({len(ord_feats)}): {ord_feats}")
    print(f"Nominal features ({len(nom_feats)}): {nom_feats}")

    # 3. preprocessing + model
    preprocessor = get_feature_pipeline(
        num_feats=num_feats,
        ord_feats=ord_feats,
        nom_feats=nom_feats,
        model_type='tree'
    )
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('smote_tomek', SMOTETomek(random_state=random_state)),
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

    # Fast training sampling safeguard
    if len(X) > 100000:
        sample_size = min(100000, len(X))
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[sample_idx]
        y = y.iloc[sample_idx]
        print(f"Sampled {sample_size} from {len(X)} records for faster training")

    # 4. split + fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=test_size,
        random_state=random_state
    )
    
    print("Training Random Forest pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Enhanced evaluation with optimal threshold
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Get evaluation results including optimal threshold
    eval_results = evaluate_with_optimal_threshold(y_test, y_proba, default_thresh=0.4)
    
    # Calculate threshold-free metrics
    auroc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    ll = log_loss(y_test, pipeline.predict_proba(X_test))
    base_rate = y_test.mean()
    
    # Print comprehensive evaluation summary
    print("\n" + "="*60)
    print("=== COMPREHENSIVE EVALUATION SUMMARY ===")
    print("="*60)
    print(f"AUROC           : {auroc:.4f}")
    print(f"AUC-PR (AP)     : {ap:.4f}")
    print(f"Log Loss        : {ll:.4f}")
    print(f"Positive rate   : {base_rate:.4f}")
    print(f"Optimal F1 threshold: {eval_results['optimal_threshold']:.4f}")
    print(f"Optimal F1 score    : {eval_results['optimal_f1']:.4f}")
    
    # Display metrics comparison
    default_m = eval_results['default_metrics']
    optimal_m = eval_results['optimal_metrics']
    
    print("\n=== THRESHOLD COMPARISON ===")
    print(f"{'Metric':<12} {'Default (0.4)':<15} {'Optimal':<15}")
    print("-" * 45)
    print(f"{'Precision':<12} {default_m['precision']:<15.4f} {optimal_m['precision']:<15.4f}")
    print(f"{'Recall':<12} {default_m['recall']:<15.4f} {optimal_m['recall']:<15.4f}")
    print(f"{'F1-Score':<12} {default_m['f1']:<15.4f} {optimal_m['f1']:<15.4f}")
    print(f"{'MCC':<12} {default_m['mcc']:<15.4f} {optimal_m['mcc']:<15.4f}")

    # === Enhanced plotting with AUC values ===
    auroc_plot, ap_plot = plot_enhanced_curves(y_test, y_proba, artifact_path, eval_results['optimal_threshold'])

    # === Unified Reporting additions ===
    artifacts_dir = os.path.dirname(artifact_path) or "."
    os.makedirs(artifacts_dir, exist_ok=True)

    _model_tag = "RF"

    # 1) Dataset summary
    export_dataset_summary(
        X_train, y_train,
        out_csv=os.path.join(artifacts_dir, "dataset_summary.csv")
    )

    # 2) Standard PR curve (unified version)
    export_pr_curve(
        model=pipeline,
        X_test=X_test,
        y_test=y_test,
        out_png=os.path.join(artifacts_dir, f"pr_curve_{_model_tag}.png"),
        title=f"PR — {_model_tag}"
    )

    # 3) Threshold sweep including optimal threshold
    threshold_sweep(
        y_true=y_test.values,
        y_proba=y_proba,
        thresholds=eval_results['thresholds_for_sweep'],  # Includes optimal threshold
        artifacts_dir=artifacts_dir,
        csv_name=f"thresholds_{_model_tag}.csv",
        out_png=os.path.join(artifacts_dir, f"thresholds_{_model_tag}.png"),
        model_label=_model_tag,
    )

    # 4) Feature importance (Top10)
    try:
        pre = pipeline.named_steps["preproc"]
        feat_all = pre.get_feature_names_out()
        sel = pipeline.named_steps.get("sel", None)
        if sel is not None and hasattr(sel, "get_support"):
            mask = sel.get_support()
            feat_names = feat_all[mask]
        else:
            feat_names = feat_all
        export_feature_importance(
            model=pipeline.named_steps["clf"],
            feature_names=list(feat_names),
            top_k=10,
            artifacts_dir=artifacts_dir,
            csv_name=f"feature_importance_top10_{_model_tag}.csv",
            out_png=os.path.join(artifacts_dir, f"feature_importance_top10_{_model_tag}.png"),
            title=f"Top 10 features — {_model_tag}",
        )
    except Exception as e:
        print(f"[WARN] Feature importance export failed: {e}")

    # 5) Error analysis using optimal threshold
    error_analysis(
        y_true=y_test.values,
        y_proba=y_proba,
        threshold=eval_results['optimal_threshold'],  # Use optimal threshold
        artifacts_dir=artifacts_dir,
        cm_png=f"confusion_matrix_{_model_tag}.png",
        rates_csv=f"error_rates_{_model_tag}.csv",
        exemplar_csv=f"error_exemplars_{_model_tag}.csv",
        X_test=X_test,
        top_k=50,
    )

    # 6) Business scenario simulation
    simulate_intervention(
        y_true=y_test.values,
        y_proba=y_proba,
        recall_targets=[0.5, 0.6, 0.7, 0.8],
        artifacts_dir=artifacts_dir,
        out_csv=f"intervention_{_model_tag}.csv",
    )

    # 6. save model
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    print(f"Model pipeline saved to {artifact_path}")
    
    # Final summary
    print(f"\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print(f"Model: Random Forest")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUC-PR: {ap:.4f}")
    print(f"Optimal F1 threshold: {eval_results['optimal_threshold']:.4f}")
    print(f"Optimal F1 score: {eval_results['optimal_f1']:.4f}")
    
    return pipeline, eval_results


if __name__ == '__main__':
    main()