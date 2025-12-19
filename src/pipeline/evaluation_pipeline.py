import json
import os
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt

from src.config_manager import load_config
from src.components.data_ingestion import DataIngestion
from src.pipeline.inference_pipeline import InferencePipeline
from src.logging_system import get_logger
from src.exceptions import NLUException

logger = get_logger()


class EvaluationPipeline:
    def __init__(self):
        try:
            logger.info("Initializing Evaluation Pipeline")

            self.config = load_config()

            self.data_ingestor = DataIngestion(
                intents_path=self.config["paths"]["intents_path"],
                eval_path=self.config["paths"]["eval_path"]
            )

            self.eval_data = self.data_ingestor.load_eval_data()

            self.inference_pipeline = InferencePipeline()

            self.artifacts_dir = "artifacts"
            os.makedirs(self.artifacts_dir, exist_ok=True)

            logger.info("Evaluation Pipeline initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize evaluation pipeline")
            raise NLUException(e)

    def _save_metrics(self, metrics: dict):
        path = os.path.join(self.artifacts_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {path}")

    def _plot_confusion_matrix(self, y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(10, 8))
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.colorbar()

        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm[i, j], ha="center", va="center")

        path = os.path.join(self.artifacts_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        logger.info(f"Confusion matrix saved to {path}")

    def run(self):
        try:
            logger.info("Starting evaluation")

            y_true = []
            y_pred = []
            predictions_log = []

            for sample in self.eval_data:
                true_intent = sample["intent"]

                for query in sample["examples"]:
                    result = self.inference_pipeline.run(query)

                    predicted_intent = result["intent"]

                    y_true.append(true_intent)
                    y_pred.append(predicted_intent)

                    predictions_log.append({
                        "text": query,
                        "true_intent": true_intent,
                        "predicted_intent": predicted_intent,
                        "confidence": result.get("confidence")
                    })

            # Metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )

            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "total_samples": len(y_true),
                "intent_distribution": dict(Counter(y_true))
            }

            # Save outputs
            self._save_metrics(metrics)

            with open(os.path.join(self.artifacts_dir, "predictions.json"), "w") as f:
                json.dump(predictions_log, f, indent=2)

            labels = sorted(list(set(y_true)))
            self._plot_confusion_matrix(y_true, y_pred, labels)

            logger.info("Evaluation completed successfully")
            return metrics

        except Exception as e:
            logger.error("Evaluation pipeline failed")
            raise NLUException(e)
