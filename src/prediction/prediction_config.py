"""
Configuration file for the unified model prediction system
Adjust paths and parameters according to your setup
"""

# ============================================================================
# MODEL PATHS CONFIGURATION
# ============================================================================

# Update these paths to match your actual model file locations
MODEL_PATHS = {
    'catboost': 'artifacts/catboost_pipeline.pkl',
    'random_forest': 'artifacts/rf_pipeline.pkl', 
    'logistic_regression': 'artifacts/logistic_final_pipeline.pkl',
    'xgboost': 'artifacts/xgb_pipeline.pkl'
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Path to your data directory
DATA_DIR = 'data/raw/anonymisedData'

# Output paths for prediction results
PREDICTION_OUTPUT_PATH = 'data/predicted/predictions_all_models.csv'
SUMMARY_OUTPUT_PATH = 'prediction_summary.csv'

# ============================================================================
# PREDICTION PARAMETERS
# ============================================================================

# Risk thresholds for analysis
RISK_THRESHOLDS = {
    'low_risk': 0.3,      # Below this = low risk
    'medium_risk': 0.5,   # Between low and high = medium risk
    'high_risk': 0.7      # Above this = very high risk
}

# Default threshold for binary classification
DEFAULT_THRESHOLD = 0.4

# ============================================================================
# ENSEMBLE CONFIGURATION
# ============================================================================

# Weights for weighted ensemble (adjust based on your model performance)
ENSEMBLE_WEIGHTS = {
    'catboost_fail_probability': 0.30,
    'xgboost_fail_probability': 0.30,
    'random_forest_fail_probability': 0.25,
    'logistic_regression_fail_probability': 0.15
}

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

# Maximum number of samples to process at once (to manage memory)
MAX_BATCH_SIZE = 50000

# Whether to enable detailed logging
VERBOSE = True

# Number of decimal places for probability outputs
PROBABILITY_PRECISION = 4

# ============================================================================
# FEATURE ENGINEERING FLAGS
# ============================================================================

# Enable/disable specific feature engineering steps
FEATURE_ENGINEERING = {
    'create_balanced_features_for_logistic': True,  # Critical for logistic regression
    'validate_feature_consistency': True,           # Check feature alignment
    'optimize_data_types': True                     # Optimize memory usage
}

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

# Expected probability ranges (for validation)
PROBABILITY_BOUNDS = {
    'min_probability': 0.0,
    'max_probability': 1.0,
    'expected_mean_range': (0.05, 0.15),  # Typical range for student failure rates
    'max_std_deviation': 0.5
}

# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

# Columns to include in the final output
OUTPUT_COLUMNS = {
    'include_original_features': False,  # Whether to include all original features
    'include_intermediate_features': False,  # Include engineered features
    'include_model_probabilities': True,  # Individual model predictions
    'include_ensemble_probabilities': True,  # Ensemble predictions
    'include_risk_categories': True  # High/Medium/Low risk labels
}

# ============================================================================
# ADVANCED CONFIGURATION
# ============================================================================

# Memory optimization settings
MEMORY_OPTIMIZATION = {
    'use_reduced_precision': True,  # Use float32 instead of float64
    'batch_processing': True,       # Process data in batches
    'clear_intermediate_data': True  # Clear intermediate results to save memory
}

# Error handling configuration
ERROR_HANDLING = {
    'continue_on_model_failure': True,  # Continue if one model fails
    'fill_failed_predictions_with_nan': True,  # Fill failed predictions with NaN
    'log_detailed_errors': VERBOSE
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_active_models():
    """Get list of models that have valid paths"""
    import os
    active_models = {}
    for model_name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            active_models[model_name] = path
        else:
            print(f"Warning: Model file not found for {model_name}: {path}")
    return active_models


def validate_config():
    """Validate the configuration settings"""
    issues = []
    
    # Check if data directory exists
    import os
    if not os.path.exists(DATA_DIR):
        issues.append(f"Data directory not found: {DATA_DIR}")
    
    # Check model files
    active_models = get_active_models()
    if not active_models:
        issues.append("No model files found with valid paths")
    
    # Validate ensemble weights
    weight_sum = sum(ENSEMBLE_WEIGHTS.values())
    if abs(weight_sum - 1.0) > 0.01:
        issues.append(f"Ensemble weights sum to {weight_sum:.3f}, should be 1.0")
    
    # Validate thresholds
    if not (0 <= DEFAULT_THRESHOLD <= 1):
        issues.append(f"Default threshold {DEFAULT_THRESHOLD} should be between 0 and 1")
    
    return issues


def print_config_summary():
    """Print a summary of the current configuration"""
    print("=" * 60)
    print("PREDICTION SYSTEM CONFIGURATION")
    print("=" * 60)
    
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Path: {PREDICTION_OUTPUT_PATH}")
    print(f"Default Threshold: {DEFAULT_THRESHOLD}")
    print(f"Max Batch Size: {MAX_BATCH_SIZE:,}")
    
    print(f"\nActive Models:")
    active_models = get_active_models()
    for model_name, path in active_models.items():
        print(f"  ✓ {model_name}: {path}")
    
    missing_models = set(MODEL_PATHS.keys()) - set(active_models.keys())
    if missing_models:
        print(f"\nMissing Models:")
        for model_name in missing_models:
            print(f"  ✗ {model_name}: {MODEL_PATHS[model_name]}")
    
    print(f"\nEnsemble Weights:")
    for model, weight in ENSEMBLE_WEIGHTS.items():
        print(f"  {model}: {weight}")
    
    print(f"\nRisk Thresholds:")
    for risk_level, threshold in RISK_THRESHOLDS.items():
        print(f"  {risk_level}: {threshold}")
    
    # Validate configuration
    issues = validate_config()
    if issues:
        print(f"\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✅ Configuration looks good!")
    
    print("=" * 60)


if __name__ == '__main__':
    print_config_summary()