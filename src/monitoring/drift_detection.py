"""
Drift Detection Module using Evidently AI
Monitors data drift, model performance degradation, and prediction drift
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
try:
    from evidently.dashboard import Dashboard
    from evidently.dashboard.tabs import DataDriftTab
except Exception:
    Dashboard = None
    DataDriftTab = None
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import (
    DataDriftTable,
    ClassificationDummyMetric,
    ClassificationQualityMetric,
)
from evidently.report import Report

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift, concept drift, and model performance degradation"""

    def __init__(self, reference_data: pd.DataFrame, feature_names: List[str]):
        """
        Initialize drift detector with reference dataset
        
        Args:
            reference_data: Historical data for comparison baseline
            feature_names: List of feature column names
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.last_check = None

    def detect_data_drift(
        self, current_data: pd.DataFrame, threshold: float = 0.1
    ) -> Dict:
        """
        Detect data drift using statistical tests
        
        Args:
            current_data: Recent production data
            threshold: Drift detection threshold (0-1)
            
        Returns:
            Dictionary with drift metrics and feature-level drift scores
        """
        report = Report(metrics=[DataDriftTable()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
        )

        drift_report = report.as_dict()
        drift_table = drift_report["metrics"][0]["result"]["drift_by_columns"]

        features_with_drift = []
        for col_name, col_info in drift_table.items():
            if col_info.get("drift_detected", False):
                features_with_drift.append({
                    "feature": col_name,
                    "drift_score": col_info.get("drift_score", 0),
                    "test_name": col_info.get("stattest_name", "unknown"),
                })

        drift_detected = len(features_with_drift) > 0
        drift_percentage = len(features_with_drift) / len(self.feature_names)

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": drift_detected,
            "drift_percentage": drift_percentage,
            "features_with_drift": features_with_drift,
            "total_features_checked": len(self.feature_names),
            "report": drift_table,
        }

        logger.info(f"Data drift detection: {drift_detected} ({drift_percentage:.1%})")
        return result

    def detect_prediction_drift(
        self, reference_predictions: np.ndarray, current_predictions: np.ndarray
    ) -> Dict:
        """
        Detect drift in model predictions distribution
        
        Args:
            reference_predictions: Historical model predictions
            current_predictions: Recent model predictions
            
        Returns:
            Dictionary with prediction drift metrics
        """
        ref_mean = np.mean(reference_predictions)
        ref_std = np.std(reference_predictions)
        curr_mean = np.mean(current_predictions)
        curr_std = np.std(current_predictions)

        # Kolmogorov-Smirnov test statistic
        from scipy.stats import ks_2samp
        ks_stat, p_value = ks_2samp(reference_predictions, current_predictions)

        drift_detected = p_value < 0.05  # 5% significance level

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": drift_detected,
            "reference_mean": float(ref_mean),
            "reference_std": float(ref_std),
            "current_mean": float(curr_mean),
            "current_std": float(curr_std),
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "mean_shift": float(abs(curr_mean - ref_mean) / (ref_std + 1e-6)),
        }

        logger.info(f"Prediction drift: {drift_detected} (KS={ks_stat:.4f}, p={p_value:.4f})")
        return result

    def detect_performance_degradation(
        self,
        reference_labels: pd.Series,
        reference_predictions: np.ndarray,
        current_labels: pd.Series,
        current_predictions: np.ndarray,
        task_type: str = "classification",
    ) -> Dict:
        """
        Detect model performance degradation
        
        Args:
            reference_labels: Ground truth labels from training
            reference_predictions: Model predictions on training data
            current_labels: Ground truth labels from production
            current_predictions: Model predictions on production data
            task_type: "classification" or "regression"
            
        Returns:
            Dictionary with performance metrics and degradation alerts
        """
        ref_df = pd.DataFrame({
            "prediction": reference_predictions,
            "target": reference_labels,
        })
        curr_df = pd.DataFrame({
            "prediction": current_predictions,
            "target": current_labels,
        })

        if task_type == "classification":
            report = Report(metrics=[ClassificationQualityMetric()])
        else:
            # For regression, use appropriate metric
            from evidently.metrics import RegressionQualityMetric
            report = Report(metrics=[RegressionQualityMetric()])

        report.run(reference_data=ref_df, current_data=curr_df)
        metrics = report.as_dict()["metrics"][0]["result"]

        # Extract key metrics
        if task_type == "classification":
            ref_metrics = metrics.get("current", {})
            performance_degraded = (
                ref_metrics.get("f1_macro", 1.0) < 0.7 or
                ref_metrics.get("roc_auc_macro", 1.0) < 0.75
            )
        else:
            ref_metrics = metrics.get("current", {})
            performance_degraded = ref_metrics.get("rmse", float("inf")) > 0.5

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "performance_degraded": performance_degraded,
            "metrics": ref_metrics,
            "report": metrics,
        }

        logger.info(f"Performance check: degraded={performance_degraded}")
        return result

    def generate_report(
        self, current_data: pd.DataFrame, predictions: np.ndarray
    ) -> str:
        """
        Generate HTML report for visualization
        
        Args:
            current_data: Current dataset
            predictions: Model predictions
            
        Returns:
            HTML report as string
        """
        if Dashboard is not None and DataDriftTab is not None:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            dashboard.run(reference_data=self.reference_data, current_data=current_data)
            return dashboard.html

        # Fallback for newer Evidently versions without dashboard module.
        report = Report(metrics=[DataDriftTable()])
        report.run(reference_data=self.reference_data, current_data=current_data)
        return json.dumps(report.as_dict())


class ABTestingFramework:
    """A/B Testing framework for model variants"""

    def __init__(self):
        self.experiments = {}

    def create_experiment(
        self,
        experiment_id: str,
        model_a_name: str,
        model_b_name: str,
        traffic_split: float = 0.5,
    ) -> Dict:
        """
        Create A/B test experiment
        
        Args:
            experiment_id: Unique experiment identifier
            model_a_name: Name of control model
            model_b_name: Name of treatment model
            traffic_split: Proportion of traffic for variant B (0-1)
            
        Returns:
            Experiment configuration
        """
        experiment = {
            "id": experiment_id,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "traffic_split": traffic_split,
            "created_at": datetime.utcnow().isoformat(),
            "metrics_a": {"samples": 0, "predictions": [], "labels": []},
            "metrics_b": {"samples": 0, "predictions": [], "labels": []},
            "status": "running",
        }
        self.experiments[experiment_id] = experiment
        logger.info(f"Created A/B test: {experiment_id}")
        return experiment

    def record_prediction(
        self,
        experiment_id: str,
        variant: str,
        prediction: float,
        label: float = None,
    ) -> None:
        """
        Record prediction for A/B test analysis
        
        Args:
            experiment_id: Experiment ID
            variant: "A" or "B"
            prediction: Model prediction
            label: Ground truth label (optional, for later evaluation)
        """
        if experiment_id not in self.experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return

        exp = self.experiments[experiment_id]
        variant_key = f"metrics_{variant.upper()}"

        if variant_key in exp:
            exp[variant_key]["samples"] += 1
            exp[variant_key]["predictions"].append(prediction)
            if label is not None:
                exp[variant_key]["labels"].append(label)

    def get_experiment_results(self, experiment_id: str) -> Dict:
        """
        Analyze A/B test results
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Statistical comparison of models
        """
        if experiment_id not in self.experiments:
            return {}

        exp = self.experiments[experiment_id]
        metrics_a = exp["metrics_a"]
        metrics_b = exp["metrics_b"]

        # Calculate basic statistics
        result = {
            "experiment_id": experiment_id,
            "status": exp["status"],
            "model_a": {
                "name": exp["model_a"],
                "samples": metrics_a["samples"],
                "avg_prediction": float(np.mean(metrics_a["predictions"]))
                if metrics_a["predictions"]
                else None,
            },
            "model_b": {
                "name": exp["model_b"],
                "samples": metrics_b["samples"],
                "avg_prediction": float(np.mean(metrics_b["predictions"]))
                if metrics_b["predictions"]
                else None,
            },
        }

        # If labels available, calculate performance metrics
        if metrics_a["labels"] and metrics_b["labels"]:
            from sklearn.metrics import roc_auc_score, f1_score

            try:
                result["model_a"]["auc"] = float(
                    roc_auc_score(metrics_a["labels"], metrics_a["predictions"])
                )
                result["model_b"]["auc"] = float(
                    roc_auc_score(metrics_b["labels"], metrics_b["predictions"])
                )

                # Statistical significance test
                from scipy.stats import chi2_contingency
                
                n_a = len(metrics_a["labels"])
                n_b = len(metrics_b["labels"])
                if n_a > 30 and n_b > 30:  # Sufficient sample size
                    # Proportions test
                    success_a = sum(np.array(metrics_a["predictions"]) > 0.5)
                    success_b = sum(np.array(metrics_b["predictions"]) > 0.5)
                    
                    result["winner"] = (
                        "B" if result["model_b"].get("auc", 0) > result["model_a"].get("auc", 0)
                        else "A"
                    )
            except Exception as e:
                logger.warning(f"Could not calculate performance metrics: {e}")

        return result

    def conclude_experiment(self, experiment_id: str, winner: str) -> Dict:
        """
        Conclude A/B test and select winning variant
        
        Args:
            experiment_id: Experiment ID
            winner: "A" or "B"
            
        Returns:
            Experiment conclusion report
        """
        if experiment_id not in self.experiments:
            return {}

        exp = self.experiments[experiment_id]
        exp["status"] = "concluded"
        exp["winner"] = winner
        exp["concluded_at"] = datetime.utcnow().isoformat()

        logger.info(f"Experiment {experiment_id} concluded. Winner: {winner}")
        return exp


# Singleton instance
drift_detector = None
ab_testing = ABTestingFramework()


def init_drift_detector(reference_data: pd.DataFrame, features: List[str]):
    """Initialize global drift detector"""
    global drift_detector
    drift_detector = DriftDetector(reference_data, features)
