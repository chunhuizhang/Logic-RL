import re
from typing import Dict, Tuple, Optional

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def normalize_answer(answer: str) -> str:
    """Normalize answer string by removing extra spaces and punctuation"""
    # Remove extra whitespace and convert to lowercase
    answer = answer.strip().lower()
    # Remove trailing punctuation
    answer = re.sub(r'[.!?]$', '', answer)
    return answer


def match_answers(ground_truth: str, model_answer: str, question_type: str) -> bool:
    """Match ground truth with model answer based on answer type"""
    gt = normalize_answer(ground_truth)
    pred = normalize_answer(model_answer)
    
    if question_type == 'YN' or question_type == 'EX':
        # return gt == pred or (gt in pred)
        yn_pattern = r'^(yes|no)'
        gt_match = re.match(yn_pattern, gt)
        pred_match = re.match(yn_pattern, pred)
        
        if gt_match and pred_match:
            print(gt_match.group(1), pred_match.group(1))
            return gt_match.group(1) == pred_match.group(1)
        return gt == pred
    
    
    elif question_type == 'HM':
        return gt == pred
    
    elif question_type == 'MC':
        return gt == pred
    
    elif question_type in ['FA', 'FO']:
        # Convert string sets to actual sets for comparison
        try:
            gt_set = set(eval(gt))
            pred_set = set(eval(pred))
            return gt_set == pred_set 
        except:
            # Normalize graph representation
            def normalize_graph(g: str) -> str:
                # Remove spaces and convert to lowercase
                g = ''.join(g.split()).lower()
                # Sort multiple paths if it's a list
                if g.startswith('[') and g.endswith(']'):
                    try:
                        paths = eval(g)
                        return str(sorted(paths))
                    except:
                        return g
                return g
            try:
                return normalize_graph(gt) == normalize_graph(pred)
            except:
                return False
    
    return gt == pred

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: int = 1,
                 answer_reward: float = 1.0):
    """Computes comprehensive score for model response."""
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")
    
    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")
    
    gt_answer = ground_truth['answer']
    gt_question_type = ground_truth['question_type']

    # Validate answer content
    answer_score = 0
    if format_correct and answer_text:
        print(f"\n[Content Validation]")
        print(f"  Expected: {normalize_answer(gt_answer)}")
        print(f"  Predicted: {normalize_answer(answer_text)}")
        
        if match_answers(gt_answer, answer_text, gt_question_type):
            answer_score = 2
            print("  Content validation: FULL MATCH")
        else:
            answer_score = -2
            print("  Content validation: MISMATCH")
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")
    
    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")
    
    return total_score

if __name__ == "__main__":
    assert match_answers('yes', 'yes, c is a node in the graph. there is an edge c->o, which means there is a directed edge from c to o', 'YN')
    assert match_answers('no', 'no, x->h is not an edge of this graph', 'YN')