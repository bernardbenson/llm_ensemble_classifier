# voter.py
from typing import List
from collections import Counter
from loguru import logger
from config import ALL_LABELS # Import the canonical list of labels

def get_voted_labels(
    all_predictions: List[List[str]],
    original_labels_from_input: List[str],
    min_votes: int
) -> List[str]:
    """
    Determines the final set of labels based on predictions and a minimum vote threshold.

    Args:
        all_predictions: A list of label lists, where each inner list is one agent's prediction.
        original_labels_from_input: The list of labels from the original input file, treated as one voter.
        min_votes: The minimum number of votes a label must receive to be included.

    Returns:
        A sorted list of unique final labels.
    """
    sets_for_voting = [list(pred_set) for pred_set in all_predictions]
    if original_labels_from_input:
        sets_for_voting.append(list(original_labels_from_input))

    if not sets_for_voting:
        return []

    label_counts = Counter()
    for prediction_set in sets_for_voting:
        # Use a set to count each valid label only once per voter
        valid_labels_in_set = set()
        for label in prediction_set:
            if label in ALL_LABELS:
                valid_labels_in_set.add(label)
            else:
                logger.warning(f"    Warning (during voting): Encountered an invalid label '{label}', which will be ignored.")
        label_counts.update(list(valid_labels_in_set))

    final_labels = [label for label, count in label_counts.items() if count >= min_votes]
    
    return sorted(list(set(final_labels)))
