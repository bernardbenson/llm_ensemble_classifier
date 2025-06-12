# orchestrator.py
import os
import sys
import json
import asyncio
import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import asdict
from collections import Counter
from loguru import logger

# Import all other modules needed 
import config
from data_models import (
    LLMPredictionResult, VotingParameters,
    ProcessingLogItem, ReportItem, FinalOutputItem, ClassificationMetrics
)
from llm_agents import get_system_prompt
from voter import get_voted_labels
# The save functions will now receive a direct filepath
from output_writer import save_final_output, save_processing_log, save_detailed_llm_report, save_classification_report
from metrics_calculator import calculate_item_metrics, generate_aggregate_report

# Configure Loguru
# This replaces the default logger with a more informative one.
# It logs to stderr by default, which is fine for console applications.
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


async def _query_agent_async(agent_config: Dict[str, Any], text_to_classify: str) -> LLMPredictionResult:
    """
    Asynchronously queries a single LLM agent using the vanilla openai library.
    It now uses the specific prompt key assigned to the agent.
    """
    agent_name = agent_config["name"]
    model_name = agent_config["model_name"]
    client = agent_config["client"]
    # Retrieve the specific prompt key for this agent, defaulting if not set.
    prompt_key = agent_config.get("prompt_key", "default")
    system_prompt = get_system_prompt(prompt_key)

    logger.info(f"Querying {agent_name} (model: {model_name}, prompt: '{prompt_key}')...")

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_classify},
            ],
            response_format={"type": "json_object"},
        )
        
        raw_content = response.choices[0].message.content
        
        try:
            parsed_json = json.loads(raw_content)
            raw_labels = parsed_json.get("labels", [])
            reasoning = parsed_json.get("reasoning", "")
            
            if not isinstance(raw_labels, list) or not all(isinstance(lbl, str) for lbl in raw_labels):
                 raise TypeError("The 'labels' field is not a list of strings.")

            valid_labels = [lbl for lbl in raw_labels if lbl in config.ALL_LABELS]

            return LLMPredictionResult(
                agent_name=agent_name, actual_model_used=model_name,
                predicted_labels=valid_labels, raw_predicted_labels=raw_labels, reasoning=reasoning,
            )
        except (json.JSONDecodeError, TypeError) as e:
            error_msg = f"Failed to parse JSON or validate structure: {e}. Raw content: '{raw_content}'"
            logger.error(f"Error with {agent_name}: {error_msg}")
            return LLMPredictionResult(
                agent_name=agent_name, actual_model_used=model_name,
                predicted_labels=[], raw_predicted_labels=[], reasoning="", error=error_msg
            )
            
    except Exception as e:
        error_msg = f"API call failed for {agent_name}"
        logger.exception(error_msg) # Use exception to log with stack trace
        return LLMPredictionResult(
            agent_name=agent_name, actual_model_used=model_name,
            predicted_labels=[], raw_predicted_labels=[], reasoning="", error=str(e)
        )

async def run_labeling_process(input_path: str, agents: List[Dict[str, Any]], voting_scheme: str):
    """
    Orchestrates the entire data labeling workflow asynchronously, writing results after each item.
    """
    if not agents:
        logger.warning("No agents available for processing. Exiting.")
        return

    # Ensure agent names are unique to prevent data merging issues
    base_name_counts = Counter(agent['name'] for agent in agents)
    current_name_indices = Counter()
    for agent in agents:
        base_name = agent['name']
        if base_name_counts[base_name] > 1:
            current_index = current_name_indices[base_name] + 1
            current_name_indices[base_name] = current_index
            agent['name'] = f"{base_name}_{current_index}"

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_dataset = json.load(f)
        logger.info(f"Successfully loaded {len(raw_dataset)} items from {input_path}")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.critical(f"Error reading or parsing input file {input_path}: {e}")
        return

    # --- Setup all output paths once, at the beginning ---
    logger.info("--- Initializing output file paths ---")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename_base = os.path.splitext(os.path.basename(input_path))[0]
    safe_base = "".join(c for c in input_filename_base if c.isalnum() or c in ('_', '-')).rstrip()

    # Define paths for all output files
    final_output_path = os.path.join(config.OUTPUT_DIR, f"{safe_base}_{voting_scheme}_{timestamp}.json")
    log_path = os.path.join(config.LOGS_DIR, f"log_{safe_base}_{timestamp}.json")
    llm_report_path = os.path.join(config.REPORTS_DIR, f"report_{safe_base}_{timestamp}.json")
    classification_report_path = os.path.join(config.REPORTS_DIR, f"classification_report_{safe_base}_{timestamp}.txt")

    # Ensure output directories exist
    for path in [final_output_path, log_path, llm_report_path, classification_report_path]:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
    # --- Initialize data containers ---
    final_output_data: List[FinalOutputItem] = []
    processing_log_data: List[ProcessingLogItem] = []
    llm_report_data: List[ReportItem] = []
    
    all_ground_truth_for_report: List[List[str]] = []
    all_ensemble_preds_for_report: List[List[str]] = []
    all_agent_preds_for_report: Dict[str, List[List[str]]] = {agent['name']: [] for agent in agents}

    total_items = len(raw_dataset)

    for i, item_data in enumerate(raw_dataset):
        title = item_data.get("title", f"Untitled Item {i+1}")
        item_id = item_data.get("id") or item_data.get("link") or f"item_{i+1}"
        logger.info(f"--- Processing item {i+1}/{total_items}: {title[:100]}... ---")

        ground_truth_labels = item_data.get("labels") or item_data.get("original_labels", [])
        text_to_classify = item_data.get("full_text", "")
        if len(text_to_classify) > config.CHAR_LIMIT_FOR_INPUT:
            logger.warning(f"Text exceeds limit, truncating to {config.CHAR_LIMIT_FOR_INPUT} chars.")
            text_to_classify = text_to_classify[:config.CHAR_LIMIT_FOR_INPUT]
        
        tasks = [_query_agent_async(agent_config, text_to_classify) for agent_config in agents]
        agent_results = await asyncio.gather(*tasks)

        if ground_truth_labels:
            logger.debug(f"Ground Truth Labels: {ground_truth_labels}")
            all_ground_truth_for_report.append(ground_truth_labels)
            
            for result in agent_results:
                if result.error:
                    logger.warning(f"Agent {result.agent_name} finished with error.")
                    all_agent_preds_for_report[result.agent_name].append([])
                    continue
                
                logger.debug(f"Agent {result.agent_name} Predictions: {result.predicted_labels}")
                metrics = calculate_item_metrics(ground_truth_labels, result.predicted_labels, config.ALL_LABELS)
                result.metrics_vs_ground_truth = metrics
                all_agent_preds_for_report[result.agent_name].append(result.predicted_labels)
                logger.debug(f"Metrics (vs GT) for {result.agent_name}: F1={metrics.f1_score:.2f}, Jaccard={metrics.jaccard_score:.2f}")
        else:
             for result in agent_results:
                logger.debug(f"Agent {result.agent_name} Predictions: {result.predicted_labels}")
             logger.warning("No ground truth labels found for this item. Skipping metrics calculation.")

        all_model_predictions: List[List[str]] = [res.predicted_labels for res in agent_results]
        num_llms = len(agents)
        has_input_voter = bool(ground_truth_labels)
        total_voters = num_llms + (1 if has_input_voter else 0)
        min_vote_threshold = 1 if voting_scheme == 'union' else (total_voters // 2) + (total_voters % 2) if total_voters > 0 else 1
        
        logger.info(f"Voting with '{voting_scheme}' scheme ({total_voters} total voters, threshold: {min_vote_threshold})...")
        final_labels = get_voted_labels(all_model_predictions, ground_truth_labels, min_vote_threshold)
        logger.info(f"-> Final Voted Labels: {final_labels}")
        
        ensemble_metrics = None
        if ground_truth_labels:
            ensemble_metrics = calculate_item_metrics(ground_truth_labels, final_labels, config.ALL_LABELS)
            all_ensemble_preds_for_report.append(final_labels)
            logger.debug(f"Ensemble Metrics (vs GT): F1={ensemble_metrics.f1_score:.2f}, Jaccard={ensemble_metrics.jaccard_score:.2f}")

        # --- Append new data to in-memory lists ---
        final_output_data.append(FinalOutputItem(
            link=item_data.get("link"), title=title, full_text=text_to_classify, labels=final_labels
        ))
        
        voting_params = VotingParameters(
            voting_scheme_used=voting_scheme, num_llm_agents_participated=num_llms,
            input_labels_participated_as_voter=has_input_voter,
            total_voters_for_majority_calc=total_voters, majority_threshold_applied=min_vote_threshold
        )
        processing_log_data.append(ProcessingLogItem(
            link=item_data.get("link"), id=item_data.get("id"), title=title, full_text=text_to_classify,
            labels=item_data.get("labels", []), original_labels=item_data.get("original_labels", []),
            tdamm_labels_majority_voted=final_labels,
            voted_labels_metrics=ensemble_metrics,
            original_input_labels_considered_in_vote=ground_truth_labels,
            llm_processing_details=agent_results, voting_parameters=voting_params
        ))
        
        llm_report_data.append(ReportItem(
            item_identifier=item_id, title=title, input_file_labels=ground_truth_labels, llm_predictions=agent_results
        ))

        # --- Write current state of all data to files ---
        logger.info(f"Writing cumulative results to files after processing item {i+1}/{total_items}.")
        save_final_output(final_output_path, final_output_data)
        save_processing_log(log_path, processing_log_data)
        save_detailed_llm_report(llm_report_path, llm_report_data)

    # --- Finalize and Save Aggregate Report (which requires all data) ---
    logger.info("--- Generating Final Aggregate Classification Report ---")
    full_report_str = f"Aggregate Classification Report for file: {input_filename_base}.json\n"
    full_report_str += f"Generated on: {datetime.datetime.now().isoformat()}\n"
    full_report_str += f"Voting Scheme Used: {voting_scheme}\n"
    
    logger.info(f"Generating report for 'Ensemble Voted Labels' ({len(all_ensemble_preds_for_report)} items)...")
    full_report_str += generate_aggregate_report(
        all_ground_truth_for_report, all_ensemble_preds_for_report, config.ALL_LABELS, "Ensemble Voted Labels"
    )
    
    for agent_name, predictions in all_agent_preds_for_report.items():
        logger.info(f"Generating report for '{agent_name}' ({len(predictions)} items)...")
        full_report_str += generate_aggregate_report(
            all_ground_truth_for_report, predictions, config.ALL_LABELS, agent_name
        )

    save_classification_report(classification_report_path, full_report_str)
    logger.success("Labeling process completed successfully.")
