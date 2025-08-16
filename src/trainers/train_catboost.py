# src/trainers/train_catboost.py
"""
Train a CatBoost pipeline on the weekly data with comprehensive ML practices.
Includes hyperparameter tuning, class imbalance handling, and advanced evaluation.
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score,
    confusion_matrix, matthews_corrcoef, log_loss
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

from src.data.load_data import load_weekly
from src.features.preprocessor import get_feature_pipeline

# Import unified reporting functions
from .unified_reporting import (
    export_dataset_summary,
    export_pr_curve,
    threshold_sweep,
    export_feature_importance,
    error_analysis,
    simulate_intervention,
)

class CatBoostTrainer:
    """
    Comprehensive CatBoost trainer with advanced ML practices
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_params = None
        self.best_model = None
        self.preprocessor = None
        self.cat_features = None
        self.feature_names_after_preprocessing = None  # Store feature names
        
    def prepare_data(self, data_dir: str, test_size: float = 0.2):
        """
        Load and prepare data with proper feature categorization
        """
        # Load data
        df = load_weekly(data_dir)
        y = df['target_fail']
        drop_cols = [
            'id_student', 'code_module', 'code_presentation',
            'week_end', 'target_fail'
        ]
        X = df.drop(columns=drop_cols)
        
        # Define feature categories
        num_feats = [
            *[c for c in X.columns if c.startswith('cum_click_') or c.startswith('sum_click_')],
            'week', 'week_start_day', 'num_of_prev_attempts', 'studied_credits',
            'course_priority_rank', 'course_progress_ratio', 'weeks_to_course_end', 
            'repeat_risk_score', 'study_load_intensity',
            'is_early_stage', 'is_mid_stage', 'is_final_stage'
        ]
        ord_feats = [c for c in X.columns if c in ('highest_education', 'imd_band', 'age_band')]
        nom_feats = [c for c in X.columns if c in ('gender', 'region', 'disability',
                                                   'age_education_interaction', 'region_deprivation_interaction')]
        
        # Feature categorization check
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
        
        # Create preprocessor for CatBoost
        self.preprocessor = get_feature_pipeline(
            num_feats=num_feats,
            ord_feats=ord_feats,
            nom_feats=nom_feats,
            model_type='catboost'  # Use the new catboost mode
        )
        
        # Store categorical feature indices for CatBoost
        # After preprocessing, categorical features will be at the end
        self.cat_features = list(range(len(num_feats), len(num_feats) + len(ord_feats) + len(nom_feats)))
        
        # Store the feature names in the order they appear after preprocessing
        # For CatBoost with get_feature_pipeline, the order is: numerical + ordinal + nominal
        self.feature_names_after_preprocessing = num_feats + ord_feats + nom_feats
        
        print(f"Stored {len(self.feature_names_after_preprocessing)} feature names for importance plot")
        print(f"First 5 stored feature names: {self.feature_names_after_preprocessing[:5]}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set: {X_train.shape}, Positive class ratio: {y_train.mean():.3f}")
        print(f"Test set: {X_test.shape}, Positive class ratio: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def objective(self, trial, X_train, y_train):
        """
        Optuna objective function for hyperparameter optimization
        """
        # Define hyperparameter search space
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'class_weights': [1, trial.suggest_float('positive_class_weight', 5, 20)],
            'random_seed': self.random_state,
            'verbose': False,
            'eval_metric': 'AUC',
            'task_type': 'CPU'
        }
        
        # Preprocess training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Create CatBoost Pool
        train_pool = Pool(
            X_train_processed, 
            y_train, 
            cat_features=self.cat_features
        )
        
        # Cross-validation with CatBoost
        model = CatBoostClassifier(**params)
        
        # Use stratified k-fold for imbalanced data
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in skf.split(X_train_processed, y_train):
            X_fold_train, X_fold_val = X_train_processed[train_idx], X_train_processed[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            fold_train_pool = Pool(X_fold_train, y_fold_train, cat_features=self.cat_features)
            fold_val_pool = Pool(X_fold_val, y_fold_val, cat_features=self.cat_features)
            
            fold_model = CatBoostClassifier(**params)
            fold_model.fit(fold_train_pool, eval_set=fold_val_pool, early_stopping_rounds=50)
            
            y_pred_proba = fold_model.predict_proba(fold_val_pool)[:, 1]
            score = average_precision_score(y_fold_val, y_pred_proba)  # AUC-PR for imbalanced data
            scores.append(score)
        
        return np.mean(scores)
    
    def tune_hyperparameters(self, X_train, y_train, n_trials: int = 100):
        """
        Hyperparameter tuning using Optuna
        """
        print("Starting hyperparameter optimization...")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        print(f"Best AUC-PR: {study.best_value:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return study
    
    def train_best_model(self, X_train, X_test, y_train, y_test):
        """
        Train the final model with best parameters
        """
        # Use best parameters or default if not tuned
        if self.best_params is None:
            params = {
                'iterations': 1000,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3,
                'class_weights': [1, 15.67],  # Based on 6% positive class
                'random_seed': self.random_state,
                'verbose': 100,
                'eval_metric': 'AUC',
                'task_type': 'CPU'
            }
        else:
            params = self.best_params.copy()
            params.update({
                'random_seed': self.random_state,
                'verbose': 100,
                'eval_metric': 'AUC',
                'task_type': 'CPU'
            })
        
        # Preprocess data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Create CatBoost Pools
        train_pool = Pool(X_train_processed, y_train, cat_features=self.cat_features)
        test_pool = Pool(X_test_processed, y_test, cat_features=self.cat_features)
        
        # Train model with early stopping
        self.best_model = CatBoostClassifier(**params)
        self.best_model.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=100,
            plot=False
        )
        
        return self.best_model
    
    def evaluate_model(self, X_test, y_test, thresholds=None):
        """
        Comprehensive model evaluation
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess test data
        X_test_processed = self.preprocessor.transform(X_test)
        test_pool = Pool(X_test_processed, y_test, cat_features=self.cat_features)
        
        # Get predictions
        y_proba = self.best_model.predict_proba(test_pool)[:, 1]
        
        print("=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        
        # Basic metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        logloss = log_loss(y_test, self.best_model.predict_proba(test_pool))
        
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print()
        
        # Find optimal threshold based on F1-score FIRST
        precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = pr_thresholds[optimal_idx]
        
        print(f"Optimal threshold (F1-maximizing): {optimal_threshold:.3f}")
        print(f"F1-score at optimal threshold: {f1_scores[optimal_idx]:.4f}")
        print()
        
        # Set default thresholds including the optimal one
        if thresholds is None:
            # Create a set of thresholds that includes the optimal one
            default_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            # Add optimal threshold and sort
            thresholds = sorted(list(set(default_thresholds + [optimal_threshold])))
        
        # Evaluation at different thresholds
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Mark if this is the optimal threshold
            thresh_label = f"{thresh:.3f}"
            if abs(thresh - optimal_threshold) < 1e-6:
                thresh_label += " (OPTIMAL)"
                
            print(f"=== THRESHOLD = {thresh_label} ===")
            print(classification_report(y_test, y_pred))
            print(f"Matthews Correlation Coefficient: {mcc:.4f}")
            print()
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'log_loss': logloss,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': f1_scores[optimal_idx],
            'y_proba': y_proba,
            'evaluation_thresholds': thresholds  # Return the thresholds used
        }
    
    def plot_evaluation_curves(self, y_test, y_proba, save_path: str):
        """
        Plot ROC and Precision-Recall curves with AUC values
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        baseline = y_test.mean()
        
        ax2.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        ax2.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})')  # Add AUC to title
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.pkl', '_evaluation_curves.png'), dpi=300, bbox_inches='tight')
        print(f"Evaluation curves saved to {save_path.replace('.pkl', '_evaluation_curves.png')}")
        print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")  # Print AUC values
        plt.show()
    
    def plot_feature_importance(self, feature_names, save_path: str, top_k: int = 20):
        """
        Plot feature importance
        """
        if self.best_model is None:
            return
        
        # Get feature importance
        importances = self.best_model.get_feature_importance()
        
        # Ensure feature_names and importances have the same length
        if len(feature_names) != len(importances):
            print(f"Warning: Feature names length ({len(feature_names)}) != importances length ({len(importances)})")
            # Adjust to the minimum length
            min_len = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_len]
            importances = importances[:min_len]
        
        # Create feature importance dataframe
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {min(top_k, len(feature_imp_df))} Feature Importance:")
        print(feature_imp_df.head(top_k)[['feature', 'importance']].to_string(index=False))
        
        # Plot top k features
        plt.figure(figsize=(10, 8))
        top_features = feature_imp_df.head(top_k)
        
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_k} Feature Importance (CatBoost)')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        importance_path = save_path.replace('.pkl', '_feature_importance.png')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {importance_path}")
        plt.show()
        
        return feature_imp_df
    
    def get_feature_names(self, X=None):
        """
        Get proper feature names after preprocessing
        """
        # First try to use stored feature names
        if self.feature_names_after_preprocessing is not None:
            print(f"Using stored feature names: {len(self.feature_names_after_preprocessing)} features")
            return self.feature_names_after_preprocessing
        
        # Fallback: reconstruct from X if available
        if X is not None:
            print("Reconstructing feature names from input data")
            num_feats = [
                *[c for c in X.columns if c.startswith('cum_click_') or c.startswith('sum_click_')],
                'week', 'week_start_day', 'num_of_prev_attempts', 'studied_credits',
                'course_priority_rank', 'course_progress_ratio', 'weeks_to_course_end', 
                'repeat_risk_score', 'study_load_intensity',
                'is_early_stage', 'is_mid_stage', 'is_final_stage'
            ]
            ord_feats = [c for c in X.columns if c in ('highest_education', 'imd_band', 'age_band')]
            nom_feats = [c for c in X.columns if c in ('gender', 'region', 'disability',
                                                       'age_education_interaction', 'region_deprivation_interaction')]
            
            # Handle uncategorized features
            all_categorized = set(num_feats + ord_feats + nom_feats)
            all_features = set(X.columns)
            uncategorized = all_features - all_categorized
            if uncategorized:
                num_feats.extend(list(uncategorized))
            
            feature_names = num_feats + ord_feats + nom_feats
            return feature_names
        
        # Last resort: generic names if preprocessor is available
        if self.preprocessor is not None:
            try:
                # Create a dummy dataframe to get the number of features
                dummy_cols = ['week', 'num_of_prev_attempts', 'studied_credits', 'course_progress_ratio',
                            'repeat_risk_score', 'highest_education', 'gender', 'region']
                dummy_data = pd.DataFrame([[0] * len(dummy_cols)], columns=dummy_cols)
                X_processed = self.preprocessor.transform(dummy_data)
                n_features = X_processed.shape[1]
                print(f"Using generic feature names for {n_features} features")
                return [f'feature_{i}' for i in range(n_features)]
            except Exception as e:
                print(f"Warning: Could not determine feature count: {e}")
                pass
        
        # Absolute fallback
        print("Using absolute fallback feature names")
        return [f'feature_{i}' for i in range(50)]  # Default assumption
    
    def save_model(self, artifact_path: str):
        """
        Save the complete pipeline
        """
        if self.best_model is None or self.preprocessor is None:
            raise ValueError("Model and/or preprocessor not trained yet!")
        
        # Create pipeline object
        pipeline_data = {
            'preprocessor': self.preprocessor,
            'model': self.best_model,
            'cat_features': self.cat_features,
            'best_params': self.best_params,
            'feature_names': self.feature_names_after_preprocessing
        }
        
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        joblib.dump(pipeline_data, artifact_path)
        print(f"Model pipeline saved to {artifact_path}")


def main(
    data_dir: str = 'data/raw/anonymisedData',
    artifact_path: str = 'artifacts/catboost_pipeline.pkl',
    test_size: float = 0.2,
    tune_hyperparams: bool = False,
    n_trials: int = 50,
    random_state: int = 42
):
    """
    Main training function
    """
    print("Starting CatBoost training with comprehensive ML practices...")
    
    # Initialize trainer
    trainer = CatBoostTrainer(random_state=random_state)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(data_dir, test_size)
    
    # === NEW: dataset summary ===
    artifacts_dir = os.path.dirname(artifact_path) or "."
    os.makedirs(artifacts_dir, exist_ok=True)

    export_dataset_summary(
        X_train, y_train,
        out_csv=os.path.join(artifacts_dir, "dataset_summary.csv")
    )

    # Hyperparameter tuning (optional)
    if tune_hyperparams:
        study = trainer.tune_hyperparameters(X_train, y_train, n_trials=n_trials)
    
    # Train final model
    model = trainer.train_best_model(X_train, X_test, y_train, y_test)
    
    # Comprehensive evaluation - this now finds optimal threshold first
    results = trainer.evaluate_model(X_test, y_test)
    
    # Generate plots
    trainer.plot_evaluation_curves(y_test, results['y_proba'], artifact_path)
    
    # Get feature names after preprocessing for importance plot
    print("Getting feature names for importance plot...")
    feature_names = trainer.get_feature_names(X_train)  # Keep X_train parameter for fallback
    print(f"Retrieved {len(feature_names)} feature names")
    print(f"First 5 feature names: {feature_names[:5]}")
    trainer.plot_feature_importance(feature_names, artifact_path)
    
    # === NEW: Unified Reporting additions ===
    _model_tag = "CB"

    # Create a wrapper for unified reporting functions
    class _PipeForUR:
        def __init__(self, preproc, model, cat_features):
            self.preproc = preproc
            self.model = model
            self.cat_features = cat_features
            
        def predict_proba(self, X):
            X_processed = self.preproc.transform(X)
            test_pool = Pool(X_processed, cat_features=self.cat_features)
            return self.model.predict_proba(test_pool)

    _pipe = _PipeForUR(trainer.preprocessor, trainer.best_model, trainer.cat_features)

    # 1) Standard PR curve (complementary to existing plots; no naming conflict)
    export_pr_curve(
        model=_pipe,
        X_test=X_test,
        y_test=y_test,
        out_png=os.path.join(artifacts_dir, f"pr_curve_{_model_tag}.png"),
        title=f"PR — {_model_tag}"
    )

    # 2) Threshold sweep including the optimal threshold
    # Use the thresholds that were evaluated (which includes optimal)
    threshold_sweep(
        y_true=y_test.values,
        y_proba=results['y_proba'],
        thresholds=results['evaluation_thresholds'],  # Use the thresholds that include optimal
        artifacts_dir=artifacts_dir,
        csv_name=f"thresholds_{_model_tag}.csv",
        out_png=os.path.join(artifacts_dir, f"thresholds_{_model_tag}.png"),
        model_label=_model_tag,
    )

    # 3) Feature importance export
    export_feature_importance(
        model=trainer.best_model,
        feature_names=feature_names,
        top_k=10,
        artifacts_dir=artifacts_dir,
        csv_name=f"feature_importance_top10_{_model_tag}.csv",
        out_png=os.path.join(artifacts_dir, f"feature_importance_top10_{_model_tag}.png"),
        title=f"Top 10 features — {_model_tag}",
    )

    # 4) Error analysis using optimal threshold
    error_analysis(
        y_true=y_test.values,
        y_proba=results['y_proba'],
        threshold=results['optimal_threshold'],  # Use the optimal threshold
        artifacts_dir=artifacts_dir,
        cm_png=f"confusion_matrix_{_model_tag}.png",
        rates_csv=f"error_rates_{_model_tag}.csv",
        exemplar_csv=f"error_exemplars_{_model_tag}.csv",
        X_test=X_test,
        top_k=50,
    )

    # 5) Business scenario simulation: different recall targets vs required intervention numbers
    simulate_intervention(
        y_true=y_test.values,
        y_proba=results['y_proba'],
        recall_targets=[0.5, 0.6, 0.7, 0.8],
        artifacts_dir=artifacts_dir,
        out_csv=f"intervention_{_model_tag}.csv",
    )

    # Save model
    trainer.save_model(artifact_path)
    
    print(f"\nTraining completed successfully!")
    print(f"Optimal threshold: {results['optimal_threshold']:.3f}")
    print(f"Optimal F1-score: {results['optimal_f1']:.4f}")
    
    return trainer, results


if __name__ == '__main__':
    trainer, results = main(
        data_dir='data/raw/anonymisedData',
        artifact_path='artifacts/catboost_pipeline.pkl',
        test_size=0.2,
        tune_hyperparams=False,
        n_trials=50,
        random_state=42
    )