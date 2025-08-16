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
    
    def evaluate_model(self, X_test, y_test, thresholds=[0.3, 0.4, 0.5,0.6,0.7]):
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
        
        # Evaluation at different thresholds
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            print(f"=== THRESHOLD = {thresh} ===")
            print(classification_report(y_test, y_pred))
            print(f"Matthews Correlation Coefficient: {mcc:.4f}")
            print()
        
        # Find optimal threshold based on F1-score
        precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = pr_thresholds[optimal_idx]
        
        print(f"Optimal threshold (F1-maximizing): {optimal_threshold:.3f}")
        print(f"F1-score at optimal threshold: {f1_scores[optimal_idx]:.4f}")
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'log_loss': logloss,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': f1_scores[optimal_idx],
            'y_proba': y_proba
        }
    
    def plot_evaluation_curves(self, y_test, y_proba, save_path: str):
        """
        Plot ROC and Precision-Recall curves
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
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.pkl', '_evaluation_curves.png'), dpi=300, bbox_inches='tight')
        print(f"Evaluation curves saved to {save_path.replace('.pkl', '_evaluation_curves.png')}")
        plt.show()
    
    def plot_feature_importance(self, feature_names, save_path: str, top_k: int = 20):
        """
        Plot feature importance
        """
        if self.best_model is None:
            return
        
        # Get feature importance
        importances = self.best_model.get_feature_importance()
        
        # Create feature importance dataframe
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
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
            'best_params': self.best_params
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
    
    # Hyperparameter tuning (optional)
    if tune_hyperparams:
        study = trainer.tune_hyperparameters(X_train, y_train, n_trials=n_trials)
    
    # Train final model
    model = trainer.train_best_model(X_train, X_test, y_train, y_test)
    
    # Comprehensive evaluation
    results = trainer.evaluate_model(X_test, y_test)
    
    # Generate plots
    trainer.plot_evaluation_curves(y_test, results['y_proba'], artifact_path)
    
    # Get feature names after preprocessing for importance plot
    X_sample = trainer.preprocessor.transform(X_train.head(1))
    feature_names = [f'feature_{i}' for i in range(X_sample.shape[1])]  # Generic names
    trainer.plot_feature_importance(feature_names, artifact_path)
    
    # Save model
    trainer.save_model(artifact_path)
    
    print("\nTraining completed successfully!")
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

