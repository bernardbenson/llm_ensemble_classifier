from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# creating dataclasses for consistent outputs
@dataclass
class ClassificationMetrics:
    """
    Holds the calcualted classification metrics for a prediction. 
    """
    precision: float
    recall: float
    f1_score: float
    hamming_loss: float
    jaccard_score: float

@dataclass
class LLMPredictionResults:
    """
    Stores the coutcome of a single LLM agent's prediction for a document
    """
    agent_name: str
    actual_model_used: str
    predicted_labels: List[str]
    raw_predicted_labels: List[str]
    reasoning: str
    error: Optional[str] = None
    metrics_vs_ground_truth: Optional[ClassificationMetrics] = None

@dataclass
class VotingParameters:
    """
    Captures the parameters used for the voting process on a single item
    """
    voting_scheme_used: str
    num_llm_agents_participated: int
    input_labels_partcipated_as_voter: bool
    total_voters_for_majority_calc: int
    majority_threshold_applied: int

@dataclass
class ProcessingLogItem:
    """
    Contains all information for a single processed item for detailed logging
    """
    # original data fields from dataset
    link: Optional[str]
    id: Optional[str]
    title: str
    full_text: str
    labels: List[str]
    original_labels: List[str]

    # Added processing fields
    tdamm_labels_majority_voted: List[str]
    voted_labels_metrics: Optional[ClassificationMetrics]
    original_input_labels_considered_in_vote: List[str]
    llm_processing_details: List[LLMPredictionResults]
    voting_parameters: VotingParameters

    @dataclass
    class ReportItem:
        """
        Structures the data for the high-level LLM performance report. 
        """
        item_identifier: str
        title: str
        input_file_labels: List[str]
        llm_predictions: List[LLMPredictionResults]

    @dataclass
    class FinalOutputItem:
        """
        Defines the structure for the simplified, final output JSON file. 
        """
        link: Optional[str]
        title: str
        full_text: str
        labels: List[str]

