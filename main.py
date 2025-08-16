#!/usr/bin/env python3
"""
Main Controller Script for ML Pipeline
Orchestrates data processing, model training, prediction, and dashboard
"""

import os
import sys
import logging
import argparse
import subprocess
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add src directory to Python path for imports
PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

class MLPipelineController:
    """Main controller for the ML pipeline"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.project_root = PROJECT_ROOT
        self.artifacts_path = self.project_root / "artifacts"
        self.results_path = self.project_root / "results" / "predictions"
        
        # Ensure directories exist
        self.artifacts_path.mkdir(exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = log_dir / log_filename
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def run_training_script(self, script_name: str) -> bool:
        """Run a single training script"""
        script_path = SRC_PATH / "trainers" / script_name
        
        if not script_path.exists():
            self.logger.error(f"Training script not found: {script_path}")
            return False
            
        self.logger.info(f"Starting training: {script_name}")
        
        try:
            # Import and run the main function
            script_module = script_name.replace('.py', '')
            
            if script_module == "train_catboost":
                from trainers.train_catboost import main
            elif script_module == "train_logistic":
                from trainers.train_logistic import main
            elif script_module == "train_rf_SMOTE":
                from trainers.train_rf_SMOTE import main
            elif script_module == "train_xgboost":
                from trainers.train_xgboost import main
            else:
                self.logger.error(f"Unknown training script: {script_module}")
                return False
                
            # Run the main function
            main()
            self.logger.info(f"Training completed successfully: {script_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed for {script_name}: {str(e)}")
            return False
    
    def train_models_sequential(self) -> Dict[str, bool]:
        """Train all models sequentially"""
        training_scripts = [
            "train_catboost.py",
            "train_logistic.py", 
            "train_rf_SMOTE.py",
            "train_xgboost.py"
        ]
        
        results = {}
        self.logger.info("Starting sequential model training...")
        
        for script in training_scripts:
            success = self.run_training_script(script)
            results[script] = success
            
            if not success:
                self.logger.warning(f"Training failed for {script}")
                user_input = input(f"Continue with remaining models? (y/n): ")
                if user_input.lower() != 'y':
                    break
        
        return results
    
    def train_models_parallel(self) -> Dict[str, bool]:
        """Train all models in parallel using multiprocessing"""
        training_scripts = [
            "train_catboost.py",
            "train_logistic.py", 
            "train_rf_SMOTE.py", 
            "train_xgboost.py"
        ]
        
        self.logger.info("Starting parallel model training...")
        
        # Use subprocess for parallel execution to avoid import conflicts
        processes = []
        results = {}
        
        for script in training_scripts:
            script_path = SRC_PATH / "trainers" / script
            cmd = [sys.executable, str(script_path)]
            
            self.logger.info(f"Starting parallel training: {script}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append((script, process))
        
        # Wait for all processes to complete
        for script, process in processes:
            stdout, stderr = process.communicate()
            success = process.returncode == 0
            results[script] = success
            
            if success:
                self.logger.info(f"Parallel training completed: {script}")
            else:
                self.logger.error(f"Parallel training failed: {script}")
                self.logger.error(f"Error output: {stderr}")
        
        return results
    
    def check_model_artifacts(self) -> List[str]:
        """Check which model artifacts are available"""
        model_files = list(self.artifacts_path.glob("*.pkl"))
        self.logger.info(f"Found {len(model_files)} model artifacts: {[f.name for f in model_files]}")
        return [f.name for f in model_files]
    
    def run_prediction(self) -> bool:
        """Run unified model prediction"""
        self.logger.info("Starting unified prediction...")
        
        # Check if model artifacts exist
        model_files = self.check_model_artifacts()
        if not model_files:
            self.logger.error("No model artifacts found. Please train models first.")
            return False
        
        try:
            from prediction.unified_model_predictor import main_prediction_pipeline
            
            # Set up paths for prediction
            data_dir = str(self.project_root / 'data' / 'raw' / 'anonymisedData')
            output_path = str(self.project_root / 'data' / 'predicted' / 'predictions_all_models.csv')
            summary_path = str(self.project_root / 'prediction_summary.csv')
            
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Running prediction with data_dir: {data_dir}")
            self.logger.info(f"Output will be saved to: {output_path}")
            
            # Run the prediction pipeline
            predictions, summary = main_prediction_pipeline(
                data_dir=data_dir,
                output_path=output_path,
                summary_path=summary_path
            )
            
            # Check if prediction output was created
            if Path(output_path).exists():
                self.logger.info(f"Prediction completed successfully. Output: {output_path}")
                return True
            else:
                self.logger.error("Prediction file not found after execution")
                return False
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return False
    
    def run_dashboard(self) -> bool:
        """Launch the dashboard"""
        self.logger.info("Starting dashboard...")
        
        # Check if prediction results exist
        prediction_file = self.project_root / "data" / "predicted" / "predictions_all_models.csv"
        if not prediction_file.exists():
            self.logger.error(f"Prediction results not found at {prediction_file}. Please run prediction first.")
            return False
        
        try:
            # Run dashboard as subprocess to avoid blocking
            dashboard_script = SRC_PATH / "dashboard" / "dashboard.py"
            self.logger.info(f"Launching dashboard from: {dashboard_script}")
            self.logger.info("Dashboard will be available at: http://127.0.0.1:8050")
            self.logger.info("Press Ctrl+C to stop the dashboard")
            
            # Use subprocess to run the dashboard script
            process = subprocess.run(
                [sys.executable, str(dashboard_script)],
                cwd=str(self.project_root)  # Set working directory to project root
            )
            
            self.logger.info("Dashboard process completed")
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Dashboard stopped by user")
            return True
        except Exception as e:
            self.logger.error(f"Dashboard launch failed: {str(e)}")
            return False
    
    def run_full_pipeline(self, parallel_training: bool = False, skip_training: bool = False):
        """Run the complete ML pipeline"""
        self.logger.info("="*60)
        self.logger.info("STARTING COMPLETE ML PIPELINE")
        self.logger.info("="*60)
        
        # Step 1: Model Training
        if not skip_training:
            self.logger.info("STEP 1: Model Training")
            if parallel_training:
                training_results = self.train_models_parallel()
            else:
                training_results = self.train_models_sequential()
            
            successful_models = sum(training_results.values())
            total_models = len(training_results)
            
            self.logger.info(f"Training Summary: {successful_models}/{total_models} models trained successfully")
            
            if successful_models == 0:
                self.logger.error("No models were trained successfully. Stopping pipeline.")
                return
            elif successful_models < total_models:
                user_input = input("Some models failed to train. Continue with available models? (y/n): ")
                if user_input.lower() != 'y':
                    return
        else:
            self.logger.info("STEP 1: Skipping model training")
        
        # Step 2: Unified Prediction
        self.logger.info("STEP 2: Unified Prediction")
        if not self.run_prediction():
            self.logger.error("Prediction failed. Stopping pipeline.")
            return
        
        # Step 3: Dashboard
        self.logger.info("STEP 3: Dashboard Launch")
        if not self.run_dashboard():
            self.logger.error("Dashboard launch failed.")
            return
        
        self.logger.info("="*60)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ML Pipeline Controller')
    parser.add_argument('--parallel', action='store_true', 
                       help='Train models in parallel (default: sequential)')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip model training and use existing artifacts')
    parser.add_argument('--only-prediction', action='store_true', 
                       help='Only run prediction (requires trained models)')
    parser.add_argument('--only-dashboard', action='store_true', 
                       help='Only launch dashboard (requires prediction results)')
    parser.add_argument('--check-artifacts', action='store_true', 
                       help='Check available model artifacts and exit')
    
    args = parser.parse_args()
    
    controller = MLPipelineController()
    
    if args.check_artifacts:
        controller.check_model_artifacts()
        return
    
    if args.only_dashboard:
        controller.run_dashboard()
        return
    
    if args.only_prediction:
        controller.run_prediction()
        return
    
    # Run full pipeline
    controller.run_full_pipeline(
        parallel_training=args.parallel,
        skip_training=args.skip_training
    )

if __name__ == "__main__":
    main()