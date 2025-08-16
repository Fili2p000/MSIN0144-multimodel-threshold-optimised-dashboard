"""
Unified Multi-Model Predictor for Student Failure Prediction
Supports CatBoost, Random Forest, Logistic Regression, and XGBoost models
Modified to include test set evaluation functionality
"""
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import your modules
from src.data.load_data import load_weekly
from src.features.preprocessor import get_feature_pipeline

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedModelPredictor:
    """
    Unified predictor for all four trained models with exact feature engineering replication
    """
    
    def __init__(self):
        self.models = {}
        self.feature_processors = {}
        
    def create_balanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Logistic Regression specific feature engineering
        Replicates the create_balanced_features function from train_logistic_final.py
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

    def get_base_features(self, X: pd.DataFrame) -> tuple:
        """
        Get the standardized feature categorization used across all models
        """
        # Base numerical features (same across all models)
        num_feats = [
            *[c for c in X.columns if c.startswith('cum_click_') or c.startswith('sum_click_')],
            'week', 'week_start_day', 'num_of_prev_attempts', 'studied_credits',
            'course_priority_rank', 'course_progress_ratio', 'weeks_to_course_end', 
            'repeat_risk_score', 'study_load_intensity',
            'is_early_stage', 'is_mid_stage', 'is_final_stage'
        ]
        
        # Ordinal features (same across all models)
        ord_feats = [c for c in X.columns if c in ('highest_education', 'imd_band', 'age_band')]
        
        # Nominal features (same across all models)
        nom_feats = [c for c in X.columns if c in ('gender', 'region', 'disability',
                                                   'age_education_interaction', 'region_deprivation_interaction')]
        
        return num_feats, ord_feats, nom_feats

    def prepare_catboost_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for CatBoost model with exact replication from trainer
        """
        logger.info("Preparing data for CatBoost model...")
        
        # Drop the same columns as in training
        drop_cols = ['id_student', 'code_module', 'code_presentation', 'week_end', 'target_fail']
        X = data.drop(columns=[col for col in drop_cols if col in data.columns])
        
        # Get feature categories
        num_feats, ord_feats, nom_feats = self.get_base_features(X)
        
        # Check for uncategorized features
        all_categorized_features = set(num_feats + ord_feats + nom_feats)
        all_features = set(X.columns)
        uncategorized_features = all_features - all_categorized_features
        
        if uncategorized_features:
            logger.warning(f"CatBoost - Uncategorized features: {uncategorized_features}")
        
        return X

    def prepare_rf_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Random Forest model with exact replication from trainer
        """
        logger.info("Preparing data for Random Forest model...")
        
        # Same preprocessing as CatBoost (they use identical feature preparation)
        return self.prepare_catboost_data(data)

    def prepare_logistic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Logistic Regression model with exact replication from trainer
        """
        logger.info("Preparing data for Logistic Regression model...")
        
        # First apply the balanced features (CRITICAL for Logistic Regression)
        data_with_features = self.create_balanced_features(data)
        
        # Then drop the same columns as in training
        drop_cols = ['id_student', 'code_module', 'code_presentation', 'week_end', 'target_fail']
        X = data_with_features.drop(columns=[col for col in drop_cols if col in data_with_features.columns])
        
        # Get base features
        num_feats, ord_feats, nom_feats = self.get_base_features(X)
        
        # Add the additional features created by create_balanced_features
        additional_num_feats = [
            'course_progress_squared', 'repeat_risk_squared',
            'progress_risk_interaction', 'progress_weeks_interaction',
            'quiz_content_ratio', 'active_passive_ratio',
            'total_engagement', 'total_engagement_log1p',
            'high_repeat_risk', 'low_progress', 'minimal_engagement', 'late_in_course'
        ]
        
        # Add log features
        log_features = [c for c in X.columns if c.endswith('_log1p')]
        additional_num_feats.extend(log_features)
        
        # Extend numerical features with additional ones
        num_feats.extend(additional_num_feats)
        
        # Handle any remaining uncategorized features
        all_categorized = set(num_feats + ord_feats + nom_feats)
        all_features = set(X.columns)
        uncategorized = all_features - all_categorized
        if uncategorized:
            logger.warning(f"Logistic - Adding uncategorized features to numerical: {uncategorized}")
            num_feats.extend(list(uncategorized))
        
        return X

    def prepare_xgboost_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for XGBoost model with exact replication from trainer
        """
        logger.info("Preparing data for XGBoost model...")
        
        # Same preprocessing as CatBoost and RF
        return self.prepare_catboost_data(data)

    def get_train_test_split(self, data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Perform the same train_test_split as used in training
        
        Args:
            data: Full dataset
            test_size: Proportion for test set (default: 0.2)
            random_state: Random seed (default: 42)
            
        Returns:
            tuple: (train_data, test_data)
        """
        logger.info(f"Performing train_test_split with test_size={test_size}, random_state={random_state}")
        
        # Check if target column exists
        if 'target_fail' not in data.columns:
            logger.warning("target_fail column not found, splitting without stratification")
            train_data, test_data = train_test_split(
                data, 
                test_size=test_size, 
                random_state=random_state
            )
        else:
            # Use stratified split to maintain class distribution
            train_data, test_data = train_test_split(
                data, 
                test_size=test_size, 
                random_state=random_state,
                stratify=data['target_fail']
            )
            
            logger.info(f"Train set size: {len(train_data)}, Test set size: {len(test_data)}")
            if 'target_fail' in data.columns:
                logger.info(f"Train set class distribution: {train_data['target_fail'].value_counts().to_dict()}")
                logger.info(f"Test set class distribution: {test_data['target_fail'].value_counts().to_dict()}")
        
        return train_data, test_data

    def load_models(self, model_paths: Dict[str, str]):
        """
        Load all models from their respective paths
        
        Args:
            model_paths: Dictionary with model names as keys and file paths as values
                       e.g., {'catboost': 'artifacts/catboost_pipeline.pkl', ...}
        """
        logger.info("Loading all models...")
        
        for model_name, model_path in model_paths.items():
            try:
                logger.info(f"Loading {model_name} from {model_path}")
                
                with open(model_path, 'rb') as f:
                    model_data = joblib.load(f)
                
                if model_name == 'catboost':
                    # CatBoost saves as dictionary with specific structure
                    if isinstance(model_data, dict):
                        self.models[model_name] = model_data
                        logger.info(f"CatBoost loaded as dictionary with keys: {list(model_data.keys())}")
                    else:
                        # Fallback if saved as pipeline
                        self.models[model_name] = model_data
                        logger.info("CatBoost loaded as pipeline")
                else:
                    # Other models save as pipeline
                    self.models[model_name] = model_data
                    logger.info(f"{model_name} loaded as pipeline")
                
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
                raise

    def predict_single_model(self, data: pd.DataFrame, model_name: str) -> np.ndarray:
        """
        Predict using a single model with exact replication of training preprocessing
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        logger.info(f"Predicting with {model_name}...")
        
        # Prepare data according to model-specific requirements
        if model_name == 'catboost':
            processed_data = self.prepare_catboost_data(data)
        elif model_name == 'random_forest':
            processed_data = self.prepare_rf_data(data)
        elif model_name == 'logistic_regression':
            processed_data = self.prepare_logistic_data(data)
        elif model_name == 'xgboost':
            processed_data = self.prepare_xgboost_data(data)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Get the model
        model = self.models[model_name]
        
        try:
            if model_name == 'catboost':
                # Handle CatBoost special case
                if isinstance(model, dict):
                    # CatBoost saved as dictionary
                    preprocessor = model['preprocessor']
                    catboost_model = model['model']
                    cat_features = model.get('cat_features', [])
                    
                    # Preprocess data
                    X_processed = preprocessor.transform(processed_data)
                    
                    # Create CatBoost Pool for prediction
                    from catboost import Pool
                    test_pool = Pool(X_processed, cat_features=cat_features)
                    
                    # Predict
                    probabilities = catboost_model.predict_proba(test_pool)
                    fail_probabilities = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.ravel()
                else:
                    # CatBoost saved as pipeline (fallback)
                    probabilities = model.predict_proba(processed_data)
                    fail_probabilities = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.ravel()
            else:
                # Other models (pipeline format)
                probabilities = model.predict_proba(processed_data)
                fail_probabilities = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.ravel()
            
            logger.info(f"{model_name} prediction completed. Samples: {len(fail_probabilities)}, "
                       f"Mean probability: {fail_probabilities.mean():.4f}")
            
            return fail_probabilities
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {str(e)}")
            raise

    def predict_all_models(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using all loaded models and add results as new columns
        """
        logger.info("Starting predictions for all models...")
        
        result_data = data.copy()
        
        for model_name in self.models.keys():
            try:
                predictions = self.predict_single_model(data, model_name)
                column_name = f'{model_name}_fail_probability'
                result_data[column_name] = predictions
                
                logger.info(f"Added column: {column_name}")
                
            except Exception as e:
                logger.error(f"Failed to predict with {model_name}: {str(e)}")
                # Add NaN column to maintain consistency
                result_data[f'{model_name}_fail_probability'] = np.nan
        
        # Add ensemble predictions
        result_data = self.add_ensemble_predictions(result_data)
        
        return result_data

    def add_ensemble_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add ensemble predictions using different strategies
        """
        prediction_columns = [col for col in data.columns if col.endswith('_fail_probability') and not col.startswith('ensemble')]
        
        if len(prediction_columns) < 2:
            logger.warning("Less than 2 models available for ensemble")
            return data
        
        logger.info(f"Creating ensemble from {len(prediction_columns)} models")
        
        # Simple average ensemble
        data['ensemble_average_fail_probability'] = data[prediction_columns].mean(axis=1)
        
        # Weighted ensemble (based on typical model performance expectations)
        weights = {
            'catboost_fail_probability': 0.3,
            'xgboost_fail_probability': 0.3, 
            'random_forest_fail_probability': 0.25,
            'logistic_regression_fail_probability': 0.15
        }
        
        weighted_sum = 0
        weight_sum = 0
        for col in prediction_columns:
            if col in data.columns and not data[col].isna().all():
                weight = weights.get(col, 0.25)  # Default weight
                weighted_sum += data[col] * weight
                weight_sum += weight
        
        if weight_sum > 0:
            data['ensemble_weighted_fail_probability'] = weighted_sum / weight_sum
        
        # Median ensemble (robust to outliers)
        data['ensemble_median_fail_probability'] = data[prediction_columns].median(axis=1)
        
        return data

    def get_prediction_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary of predictions across all models
        """
        prob_columns = [col for col in data.columns if 'fail_probability' in col]
        
        if not prob_columns:
            logger.warning("No probability columns found")
            return pd.DataFrame()
        
        summary_stats = []
        
        for col in prob_columns:
            if not data[col].isna().all():
                stats = {
                    'model': col.replace('_fail_probability', ''),
                    'mean_probability': data[col].mean(),
                    'median_probability': data[col].median(),
                    'std_probability': data[col].std(),
                    'min_probability': data[col].min(),
                    'max_probability': data[col].max(),
                    'high_risk_count': (data[col] > 0.5).sum(),
                    'medium_risk_count': ((data[col] > 0.3) & (data[col] <= 0.5)).sum(),
                    'low_risk_count': (data[col] <= 0.3).sum()
                }
                summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        return summary_df.round(4)

    def evaluate_test_set_performance(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics for test set predictions (if ground truth is available)
        """
        if 'target_fail' not in test_data.columns:
            logger.warning("Ground truth (target_fail) not available for performance evaluation")
            return pd.DataFrame()
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        prob_columns = [col for col in test_data.columns if 'fail_probability' in col]
        
        if not prob_columns:
            logger.warning("No probability columns found for evaluation")
            return pd.DataFrame()
        
        y_true = test_data['target_fail']
        metrics_results = []
        
        for col in prob_columns:
            if not test_data[col].isna().all():
                y_prob = test_data[col]
                y_pred = (y_prob > 0.5).astype(int)  # Binary classification threshold
                
                try:
                    metrics = {
                        'model': col.replace('_fail_probability', ''),
                        'accuracy': accuracy_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred, zero_division=0),
                        'recall': recall_score(y_true, y_pred, zero_division=0),
                        'f1_score': f1_score(y_true, y_pred, zero_division=0),
                        'roc_auc': roc_auc_score(y_true, y_prob)
                    }
                    metrics_results.append(metrics)
                except Exception as e:
                    logger.error(f"Error calculating metrics for {col}: {str(e)}")
        
        if metrics_results:
            metrics_df = pd.DataFrame(metrics_results)
            return metrics_df.round(4)
        else:
            return pd.DataFrame()


def main_prediction_pipeline(
    data_dir: str = 'data/raw/anonymisedData',
    model_paths: Dict[str, str] = None,
    output_path: str = 'predictions_all_models.csv',
    summary_path: str = 'prediction_summary.csv'
):
    """
    Main function to run the complete prediction pipeline on full dataset
    """
    # Default model paths
    if model_paths is None:
        model_paths = {
            'catboost': 'artifacts/catboost_pipeline.pkl',
            'random_forest': 'artifacts/rf_pipeline.pkl',
            'logistic_regression': 'artifacts/logistic_final_pipeline.pkl',
            'xgboost': 'artifacts/xgb_pipeline.pkl'
        }
    
    logger.info("Starting unified model prediction pipeline...")
    
    # 1. Initialize predictor
    predictor = UnifiedModelPredictor()
    
    # 2. Load models
    predictor.load_models(model_paths)
    
    # 3. Load original data
    logger.info(f"Loading data from {data_dir}")
    original_data = load_weekly(data_dir)
    logger.info(f"Loaded {len(original_data)} samples")
    
    # 4. Run predictions
    predictions_data = predictor.predict_all_models(original_data)
    
    # 5. Generate summary
    summary = predictor.get_prediction_summary(predictions_data)
    
    # 6. Save results
    logger.info(f"Saving predictions to {output_path}")
    predictions_data.to_csv(output_path, index=False)
    
    logger.info(f"Saving summary to {summary_path}")
    summary.to_csv(summary_path, index=False)
    
    # 7. Display summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    
    # 8. Show correlation between models
    prob_columns = [col for col in predictions_data.columns if 'fail_probability' in col]
    if len(prob_columns) > 1:
        print(f"\n" + "="*80)
        print("MODEL CORRELATION MATRIX")
        print("="*80)
        correlation_matrix = predictions_data[prob_columns].corr()
        print(correlation_matrix.round(4).to_string())
    
    logger.info("Prediction pipeline completed successfully!")
    
    return predictions_data, summary


def test_set_evaluation_pipeline(
    data_dir: str = 'data/raw/anonymisedData',
    model_paths: Dict[str, str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    output_path: str = 'test_set_predictions.csv',
    summary_path: str = 'test_set_prediction_summary.csv',
    metrics_path: str = 'test_set_performance_metrics.csv'
):
    """
    NEW FUNCTION: Evaluate models on the exact same test set used in training
    """
    # Default model paths
    if model_paths is None:
        model_paths = {
            'catboost': 'artifacts/catboost_pipeline.pkl',
            'random_forest': 'artifacts/rf_pipeline.pkl',
            'logistic_regression': 'artifacts/logistic_final_pipeline.pkl',
            'xgboost': 'artifacts/xgb_pipeline.pkl'
        }
    
    logger.info("="*80)
    logger.info("STARTING TEST SET EVALUATION PIPELINE")
    logger.info("="*80)
    
    # 1. Initialize predictor
    predictor = UnifiedModelPredictor()
    
    # 2. Load models
    predictor.load_models(model_paths)
    
    # 3. Load original data
    logger.info(f"Loading data from {data_dir}")
    original_data = load_weekly(data_dir)
    logger.info(f"Loaded {len(original_data)} samples")
    
    # 4. Perform the SAME train_test_split as used in training
    train_data, test_data = predictor.get_train_test_split(
        original_data, 
        test_size=test_size, 
        random_state=random_state
    )
    
    logger.info(f"Using ONLY the test set for evaluation: {len(test_data)} samples")
    
    # 5. Run predictions on test set
    test_predictions = predictor.predict_all_models(test_data)
    
    # 6. Generate summary for test set
    test_summary = predictor.get_prediction_summary(test_predictions)
    
    # 7. Calculate performance metrics (if ground truth available)
    performance_metrics = predictor.evaluate_test_set_performance(test_predictions)
    
    # 8. Save results
    logger.info(f"Saving test set predictions to {output_path}")
    test_predictions.to_csv(output_path, index=False)
    
    logger.info(f"Saving test set summary to {summary_path}")
    test_summary.to_csv(summary_path, index=False)
    
    if not performance_metrics.empty:
        logger.info(f"Saving performance metrics to {metrics_path}")
        performance_metrics.to_csv(metrics_path, index=False)
    
    # 9. Display results
    print("\n" + "="*80)
    print("TEST SET PREDICTION SUMMARY")
    print("="*80)
    print(test_summary.to_string(index=False))
    
    if not performance_metrics.empty:
        print(f"\n" + "="*80)
        print("TEST SET PERFORMANCE METRICS")
        print("="*80)
        print(performance_metrics.to_string(index=False))
    
    # 10. Show correlation between models on test set
    prob_columns = [col for col in test_predictions.columns if 'fail_probability' in col]
    if len(prob_columns) > 1:
        print(f"\n" + "="*80)
        print("MODEL CORRELATION MATRIX (TEST SET)")
        print("="*80)
        correlation_matrix = test_predictions[prob_columns].corr()
        print(correlation_matrix.round(4).to_string())
    
    logger.info("Test set evaluation pipeline completed successfully!")
    
    return test_predictions, test_summary, performance_metrics


if __name__ == '__main__':
    # Example usage - you can choose which pipeline to run
    
    # # Option 1: Run prediction on full dataset (original functionality)
    # print("Option 1: Full dataset prediction")
    # predictions, summary = main_prediction_pipeline(
    #     data_dir='data/raw/anonymisedData',
    #     output_path='data/predicted/predictions_all_models.csv',
    #     summary_path='prediction_summary.csv'
    # )
    
    print("\n" + "="*100)
    
    # Option 2: Run evaluation on test set only (NEW functionality)
    print("Option 2: Test set evaluation")
    test_predictions, test_summary, test_metrics = test_set_evaluation_pipeline(
        data_dir='data/raw/anonymisedData',
        test_size=0.2,  # Same as used in training
        random_state=42,  # Same as used in training
        output_path='data/predicted/test_set_predictions.csv',
        summary_path='test_set_prediction_summary.csv',
        metrics_path='test_set_performance_metrics.csv'
    )