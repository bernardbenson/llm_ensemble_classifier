# metrics_calculator.py
from typing import List
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, jaccard_score, precision_recall_fscore_support
import numpy as np
import pandas as pd

from data_models import ClassificationMetrics

def calculate_item_metrics(
    true_labels: List[str], pred_labels: List[str], all_possible_labels: List[str]
) -> ClassificationMetrics:
    """
    Calculates classification metrics for a single item.
    Uses 'samples' average for precision, recall, and F1 to report per-item scores.
    """
    if not true_labels and not pred_labels:
        return ClassificationMetrics(
            precision=1.0, recall=1.0, f1_score=1.0, hamming_loss=0.0, jaccard_score=1.0
        )
    
    if not all_possible_labels:
        # Fallback if the master list of labels is somehow empty
        all_possible_labels = sorted(list(set(true_labels + pred_labels)))
        
    mlb = MultiLabelBinarizer(classes=all_possible_labels)
    
    y_true = mlb.fit_transform([true_labels])
    y_pred = mlb.transform([pred_labels])

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='samples', zero_division=0)
    
    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        hamming_loss=hamming_loss(y_true, y_pred),
        jaccard_score=jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    )

def generate_aggregate_report(
    all_true_labels: List[List[str]],
    all_pred_labels: List[List[str]],
    all_possible_labels: List[str],
    model_name: str
) -> str:
    """
    Generates a comprehensive, scikit-learn style classification report for the entire dataset.
    """
    if not all_true_labels:
        return f"\n--- No ground truth labels were found for {model_name}. Cannot generate report. ---\n"
        
    mlb = MultiLabelBinarizer(classes=all_possible_labels)
    
    # Fit on all possible labels to ensure consistent encoding
    mlb.fit([all_possible_labels])

    y_true = mlb.transform(all_true_labels)
    y_pred = mlb.transform(all_pred_labels)

    # Generate the main report, setting zero_division to 0 to suppress warnings
    report = classification_report(
        y_true, y_pred, target_names=mlb.classes_, zero_division=0
    )

    # Calculate overall metrics, also setting zero_division to 0
    h_loss = hamming_loss(y_true, y_pred)
    j_score_samples = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    j_score_macro = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Construct the final report string
    full_report = f"\n{'='*25} Classification Report for: {model_name.upper()} {'='*25}\n"
    full_report += report
    full_report += f"\n--- Overall Metrics ---\n"
    full_report += f"Jaccard Score (Samples Avg): {j_score_samples:.4f}\n"
    full_report += f"Jaccard Score (Macro Avg):   {j_score_macro:.4f}\n"
    full_report += f"Hamming Loss (fraction of labels that are incorrectly predicted): {h_loss:.4f}\n"
    full_report += f"{'='* (68 + len(model_name))}\n"

    return full_report