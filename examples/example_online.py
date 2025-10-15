"""
Example usage of DeepThinkLLM in online mode - processes a single question

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import pickle
import argparse
from datetime import datetime
from deepconf import DeepThinkLLM
from vllm import SamplingParams
import time 
from dynasor.core.evaluator import math_equal

# ============= PROMPT PREPARATION FUNCTIONS =============

def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek") -> str:
    """Prepare prompt for a single question"""
    if model_type == "deepseek":
        # Format prompt using chat template for DeepSeek
        messages = [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": question}
        ]
    else:
        # Format for GPT-like models
        messages = [
            {"role": "user", "content": question}
        ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return full_prompt


def prepare_prompt_gpt(question: str, tokenizer, reasoning_effort: str = "high") -> str:
    """Prepare prompt for GPT models with reasoning effort"""
    messages = [
        {"role": "user", "content": question}
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True
    )
    
    return full_prompt


def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def evaluate_voting_results(voting_results, ground_truth):
    """Evaluate voting results against ground truth"""
    evaluation = {}
    
    for method, result in voting_results.items():
        if "top10" in method:
            continue
        if result and result.get('answer'):
            try:
                is_correct = equal_func(result['answer'], ground_truth)
            except:
                is_correct = str(result['answer']) == str(ground_truth)
            
            evaluation[method] = {
                'answer': result['answer'],
                'is_correct': is_correct,
                'confidence': result.get('confidence'),
                'num_votes': result.get('num_votes', 0)
            }
        else:
            evaluation[method] = {
                'answer': None,
                'is_correct': False,
                'confidence': None,
                'num_votes': 0
            }
    
    return evaluation


def evaluate_confidence_methods(result, ground_truth):
    """Evaluate different confidence-based methods"""
    confidence_evaluation = {}
    
    if result.mode != "online":
        return confidence_evaluation
    
    # Evaluate warmup traces by confidence threshold
    if result.warmup_traces and result.conf_bar is not None:
        warmup_above_threshold = [
            trace for trace in result.warmup_traces 
            if trace.get('min_conf', 0) >= result.conf_bar and trace.get('extracted_answer')
        ]
        
        if warmup_above_threshold:
            correct_above = sum(1 for trace in warmup_above_threshold 
                              if equal_func(trace['extracted_answer'], ground_truth))
            confidence_evaluation['warmup_above_threshold'] = {
                'total': len(warmup_above_threshold),
                'correct': correct_above,
                'accuracy': correct_above / len(warmup_above_threshold)
            }

    
    # Evaluate final traces (excluding early stopped)
    if result.final_traces:
        final_completed = [
            trace for trace in result.final_traces 
            if trace.get('stop_reason') != 'gconf_threshold' and trace.get('extracted_answer')
        ]
        
        if final_completed:
            correct_final = sum(1 for trace in final_completed 
                              if equal_func(trace['extracted_answer'], ground_truth))
            confidence_evaluation['final_completed'] = {
                'total': len(final_completed),
                'correct': correct_final,
                'accuracy': correct_final / len(final_completed)
            }
        
        # Early stopped traces
        early_stopped = [
            trace for trace in result.final_traces 
            if trace.get('stop_reason') == 'gconf_threshold' and trace.get('extracted_answer')
        ]
        
        if early_stopped:
            correct_stopped = sum(1 for trace in early_stopped 
                                if equal_func(trace['extracted_answer'], ground_truth))
            confidence_evaluation['early_stopped'] = {
                'total': len(early_stopped),
                'correct': correct_stopped,
                'accuracy': correct_stopped / len(early_stopped)
            }
    
    return confidence_evaluation


def print_evaluation_report(question, ground_truth, evaluation, confidence_eval, result):
    """Print detailed evaluation report"""
    print(f"\n=== Evaluation Report ===")
    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Confidence threshold: {result.conf_bar:.3f}")
    print(f"Warmup traces: {len(result.warmup_traces)}")
    print(f"Final traces: {len(result.final_traces)}")
    print(f"Total tokens: {result.total_tokens} (warmup: {result.warmup_tokens}, final: {result.final_tokens})")
    print(f"Total time: {result.total_time:.2f}s")
    
    # Confidence-based evaluation
    if confidence_eval:
        print(f"\n=== Confidence-Based Evaluation ===")
        print("-" * 60)
        print(f"{'Method':<20} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
        print("-" * 60)
        
        for method, stats in confidence_eval.items():
            total = stats['total']
            correct = stats['correct']
            accuracy = stats['accuracy']
            method_name = method.replace('_', ' ').title()
            print(f"{method_name:<20} {total:<8} {correct:<8} {accuracy:<10.1%}")
    
    # Count individual trace accuracy
    correct_traces = sum(1 for trace in result.all_voting_traces 
                        if trace.get('extracted_answer') and 
                        equal_func(trace['extracted_answer'], ground_truth))
    total_valid_traces = sum(1 for trace in result.all_voting_traces if trace.get('extracted_answer'))
    
    if total_valid_traces > 0:
        trace_accuracy = correct_traces / total_valid_traces
        print(f"\nOverall trace accuracy: {correct_traces}/{total_valid_traces} ({trace_accuracy:.1%})")
    
    print(f"\n=== Voting Method Results ===")
    print("-" * 80)
    print(f"{'Method':<25} {'Answer':<20} {'Correct':<8} {'Confidence':<12} {'Votes':<6}")
    print("-" * 80)
    
    correct_methods = []
    for method, eval_result in evaluation.items():
        answer = str(eval_result['answer'])[:18] + '...' if len(str(eval_result['answer'])) > 20 else str(eval_result['answer'])
        is_correct = eval_result['is_correct']
        confidence = eval_result['confidence']
        num_votes = eval_result['num_votes']
        
        correct_str = '✓' if is_correct else '✗'
        conf_str = f"{confidence:.3f}" if confidence is not None else '-'
        
        print(f"{method:<25} {answer:<20} {correct_str:<8} {conf_str:<12} {num_votes:<6}")
        
        if is_correct:
            correct_methods.append(method)
    
    print(f"\nCorrect voting methods: {correct_methods}")
    
    # Find best method by confidence among correct ones
    correct_evals = {method: eval_result for method, eval_result in evaluation.items() 
                    if eval_result['is_correct']}
    
    if correct_evals:
        best_method = max(correct_evals.items(), 
                         key=lambda x: x[1]['confidence'] if x[1]['confidence'] is not None else 0)
        print(f"Best correct method: {best_method[0]} (confidence: {best_method[1]['confidence']:.3f})")
    
    # Method performance summary
    total_methods = len(evaluation)
    correct_count = len(correct_methods)
    print(f"Method accuracy: {correct_count}/{total_methods} ({correct_count/total_methods:.1%})")


def main():
    parser = argparse.ArgumentParser(description='DeepThinkLLM Online Mode Example')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                       help='Model path or name')
    parser.add_argument('--dataset', type=str, default="brumo_2025.jsonl",
                       help='Dataset file path')
    parser.add_argument('--qid', type=int, required=True,
                       help='Question ID to process (0-based index)')
    parser.add_argument('--rid', type=str, default="online_run",
                       help='Run ID for identification')
    parser.add_argument('--warmup_traces', type=int, default=16,
                       help='Number of warmup traces')
    parser.add_argument('--total_budget', type=int, default=256,
                       help='Total trace budget')
    parser.add_argument('--confidence_percentile', type=int, default=10,
                       help='Confidence percentile for threshold')
    parser.add_argument('--window_size', type=int, default=2048,
                       help='Sliding window size for confidence computation')
    parser.add_argument('--max_tokens', type=int, default=64000,
                       help='Maximum tokens per generation')
    parser.add_argument('--model_type', type=str, default="deepseek", choices=["deepseek", "gpt"],
                       help='Model type for prompt formatting')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=0,
                       help='Top-k sampling parameter')
    parser.add_argument('--output_dir', type=str, default="outputs",
                       help='Output directory for results')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size for model')
    parser.add_argument('--no_multiple_voting', action='store_true',
                       help='Disable multiple voting analysis')
    parser.add_argument('--adaptive_threshold', action='store_true',
                       help='Adaptive threshold or not')
    parser.add_argument('--adaptive_interval', type=int, default=2048,
                       help='Interval for adaptive threshold adjustment')
    parser.add_argument('--threshold_adjustment_factor', type=float, default=0.15,
                       help='Threshold adjustment factor')
    parser.add_argument('--smoothing_alpha', type=float, default=0.3,
                       help='EMA smoothing alpha for confidence')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    
    # Validate question ID
    if args.qid >= len(data) or args.qid < 0:
        raise ValueError(f"Question ID {args.qid} is out of range (0-{len(data)-1})")
    
    question_data = data[args.qid]
    question = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()
    
    print(f"Processing question {args.qid}: {question[:100]}...")
    
    # Initialize DeepThinkLLM
    deep_llm = DeepThinkLLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True)

    # Prepare prompt
    print("Preparing prompt...")
    if args.model_type == "gpt":
        prompt = prepare_prompt_gpt(question, deep_llm.tokenizer)
    else:
        prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20,
    )
    
    # Run deep thinking in online mode
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="online",
        warmup_traces=args.warmup_traces,
        total_budget=args.total_budget,
        confidence_percentile=args.confidence_percentile,
        window_size=args.window_size,
        compute_multiple_voting=not args.no_multiple_voting,
        sampling_params=sampling_params,
        adaptive_threshold=args.adaptive_threshold,  # Enable adaptive thresholds
        adaptive_interval=args.adaptive_interval,  # Adjust every 500 tokens
        threshold_adjustment_factor=0.15,  # 15% adjustment magnitude
        smoothing_alpha=0.3,  # EMA smoothing
    )
    
    # Evaluate results against ground truth
    evaluation = None
    confidence_eval = None
    
    if ground_truth:
        if result.voting_results:
            evaluation = evaluate_voting_results(result.voting_results, ground_truth)
        
        confidence_eval = evaluate_confidence_methods(result, ground_truth)
        print_evaluation_report(question, ground_truth, evaluation, confidence_eval, result)
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = result.to_dict()
    result_data.update({
        'question': question,
        'ground_truth': ground_truth,
        'qid': args.qid,
        'run_id': args.rid,
        'evaluation': evaluation,
        'confidence_evaluation': confidence_eval
    })
    
    result_filename = f"{args.output_dir}/deepthink_online_qid{args.qid}_rid{args.rid}_{timestamp}.pkl"
    
    with open(result_filename, 'wb') as f:
        pickle.dump(result_data, f)
    
    print(f"\nResults saved to {result_filename}")


if __name__ == "__main__":
    main()