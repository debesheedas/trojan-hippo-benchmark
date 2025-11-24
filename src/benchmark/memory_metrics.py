"""
Evaluation metrics for memory agent benchmarking.
Based on MemoryAgentBench evaluation metrics.
"""

import string
import re
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import Counter

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. ROUGE metrics will not be calculated.")


def normalize_answer(answer_text: str) -> str:
    """
    Normalize text for evaluation by removing articles, punctuation, and extra whitespace.
    
    Args:
        answer_text: The text to normalize
        
    Returns:
        Normalized text string
    """
    text = answer_text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def f1_score(prediction: str, ground_truth: str) -> tuple:
    """
    Calculate F1 score between prediction and ground truth.
    
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        Tuple of (f1_score, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # Handle special cases for yes/no/noanswer responses
    special_answers = {'yes', 'no', 'noanswer'}
    if ((normalized_prediction in special_answers or normalized_ground_truth in special_answers) and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    # Tokenize both texts and calculate token overlap
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common_tokens = sum(common_tokens.values())
    
    if num_common_tokens == 0:
        return ZERO_METRIC
    
    # Calculate precision, recall, and F1
    precision = num_common_tokens / len(prediction_tokens)
    recall = num_common_tokens / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """
    Check if prediction is an exact match with ground truth after normalization.
    
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        Boolean indicating exact match
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction: str, ground_truth: str) -> bool:
    """
    Check if ground truth is a substring of the prediction after normalization.
    
    Args:
        prediction: The predicted text  
        ground_truth: The ground truth text
        
    Returns:
        Boolean indicating substring match
    """
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def metric_max_over_ground_truths(
    metric_function,
    prediction: str,
    ground_truths: Union[str, List[str], List[List[str]]]
) -> Union[float, bool]:
    """
    Calculate the maximum score over multiple ground truth answers.
    
    Args:
        metric_function: Function to calculate score between prediction and single ground truth
        prediction: The predicted text
        ground_truths: Ground truth answer(s) - can be string, list, or nested list
        
    Returns:
        Maximum score across all ground truths
    """
    # Normalize ground_truths to a flat list of strings
    if isinstance(ground_truths, str):
        ground_truth_list = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        # Flatten nested lists
        ground_truth_list = [gt for gt_sublist in ground_truths for gt in gt_sublist]
    else:
        ground_truth_list = ground_truths

    # Calculate score for each ground truth and return maximum
    return max(metric_function(prediction, gt) for gt in ground_truth_list)


def calculate_rouge_scores(prediction: str, ground_truths: Union[str, List[str]]) -> Dict[str, float]:
    """
    Calculate ROUGE scores for prediction against ground truth(s).
    
    Args:
        prediction: The predicted text
        ground_truths: Ground truth answer(s)
        
    Returns:
        Dictionary of ROUGE scores
    """
    if not ROUGE_AVAILABLE:
        return {}
    
    # Normalize ground truths to list
    if isinstance(ground_truths, str):
        answer_list = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        answer_list = [answer for answer_sublist in ground_truths for answer in answer_sublist]
    else:
        answer_list = ground_truths
    
    # Initialize ROUGE scorer
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)
    
    # Calculate ROUGE scores
    rouge_scores = [rouge_scorer_instance.score(target=answer, prediction=prediction) for answer in answer_list]
    
    # Extract maximum ROUGE metrics
    metrics = {}
    for rouge_type in rouge_scorer_instance.rouge_types:
        metrics[rouge_type + "_f1"] = max(score[rouge_type].fmeasure for score in rouge_scores)
        metrics[rouge_type + "_recall"] = max(score[rouge_type].recall for score in rouge_scores)
        metrics[rouge_type + "_precision"] = max(score[rouge_type].precision for score in rouge_scores)
    
    return metrics


def parse_output(output_text: str, answer_prefix: str = "Answer:") -> Optional[str]:
    """
    Parse model output to extract the answer portion.
    This matches the MemoryAgentBench implementation exactly.
    
    Args:
        output_text: The complete model output
        answer_prefix: The prefix that indicates where the answer starts
        
    Returns:
        Extracted answer text or None if not found
    """
    # Try multiple patterns to extract the answer
    extraction_patterns = [
        re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), 
        re.compile(r"(?:^)(.*)(?:\n|$)")
    ]
    
    for pattern in extraction_patterns:
        match = pattern.search(output_text)
        if match:
            extracted_text = match[1].strip()
            # Remove prefix again in case it was repeated
            clean_answer = re.sub(f'^{re.escape(answer_prefix)}', '', extracted_text, flags=re.IGNORECASE).strip()
            return clean_answer
    
    # Should rarely reach here, but return None if no pattern matches
    return None


def calculate_all_metrics(
    prediction: str,
    ground_truth: Union[str, List[str], List[List[str]]]
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for prediction evaluation.
    This matches MemoryAgentBench's calculate_metrics function.
    
    Args:
        prediction: The predicted text
        ground_truth: Ground truth answer(s) - can be string, list, or nested list
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {
        "exact_match": metric_max_over_ground_truths(exact_match_score, prediction, ground_truth),
        "f1": metric_max_over_ground_truths(lambda x, y: f1_score(x, y)[0], prediction, ground_truth),
        "substring_exact_match": metric_max_over_ground_truths(substring_exact_match_score, prediction, ground_truth)
    }
    
    # Add ROUGE scores if available
    rouge_metrics = calculate_rouge_scores(prediction, ground_truth)
    metrics.update(rouge_metrics)
    
    return metrics


def default_post_process(
    output: Dict[str, Any],
    answer: Union[str, List[str], List[List[str]]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Default post-processing function for model outputs.
    This matches MemoryAgentBench's default_post_process function.
    Tries both raw output and parsed output, taking the maximum scores.
    
    Args:
        output: Model output dictionary with "output" key
        answer: Ground truth answer(s)
        
    Returns:
        Tuple of (metrics_dict, additional_info_dict)
    """
    prediction = output.get("output", "")
    metrics = calculate_all_metrics(prediction, answer)
    
    # Try parsing output and take maximum scores
    parsed_prediction = parse_output(prediction)
    if parsed_prediction is not None:
        parsed_metrics = calculate_all_metrics(parsed_prediction, answer)
        metrics = {metric_name: max(original_score, parsed_metrics[metric_name]) 
                  for metric_name, original_score in metrics.items()}
    
    return metrics, {"parsed_output": parsed_prediction}


def evaluate_response(
    agent_response: str,
    ground_truth: Union[str, List[str], List[List[str]]],
    query: Optional[str] = None,
    use_post_process: bool = True
) -> Dict[str, Any]:
    """
    Evaluate an agent response against ground truth.
    Uses MemoryAgentBench's default_post_process by default.
    
    Args:
        agent_response: The agent's response text
        ground_truth: Ground truth answer(s)
        query: Optional query text for context
        use_post_process: Whether to use post-processing (tries parsed output)
        
    Returns:
        Dictionary with metrics and evaluation results
    """
    if use_post_process:
        # Use MemoryAgentBench's default_post_process approach
        output_dict = {"output": agent_response}
        metrics, additional_info = default_post_process(output_dict, ground_truth)
    else:
        # Direct calculation without post-processing
        metrics = calculate_all_metrics(agent_response, ground_truth)
        additional_info = {}
    
    result = {
        "prediction": agent_response,
        "ground_truth": ground_truth,
        "metrics": metrics,
        "query": query,
        **additional_info
    }
    
    return result


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        Dictionary with averaged metrics
    """
    if not results:
        return {}
    
    # Collect all metric values
    metric_values = {}
    for result in results:
        metrics = result.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if metric_name not in metric_values:
                metric_values[metric_name] = []
            metric_values[metric_name].append(float(metric_value))
    
    # Calculate averages
    aggregated = {}
    for metric_name, values in metric_values.items():
        aggregated[metric_name] = sum(values) / len(values)
    
    return aggregated

