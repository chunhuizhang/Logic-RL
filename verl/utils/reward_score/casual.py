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
            return gt, pred, gt_match.group(1) == pred_match.group(1)
        return gt, pred, gt == pred
    
    elif question_type == 'HM':
        # Extract numbers from answers for How Many questions
        gt_numbers = re.findall(r'\d+', gt)
        pred_numbers = re.findall(r'\d+', pred)

        if gt_numbers and pred_numbers:
            return gt_numbers, pred_numbers, gt_numbers[0] == pred_numbers[0]  # Compare the first number found
        
        return gt, pred, gt == pred
    
    elif question_type == 'MC':
        # Extract choice letter/number for multiple choice questions
        mc_pattern = r'^([a-f]|[1-6])'  # Match a-e or 1-5 at the beginning
        gt_match = re.match(mc_pattern, gt)
        pred_match = re.match(mc_pattern, pred)
        
        # If both contain choice identifiers, compare them
        if gt_match and pred_match:
            return gt, pred, gt_match.group(1) == pred_match.group(1)
            
        # Check if the answer is the full choice identifier (e.g., "b.")
        pred_option = re.match(r'^([a-f]|[1-6])[.)]', pred)
        if gt_match and pred_option:
            return gt, pred, gt_match.group(1) == pred_option.group(1)
            
        # Fall back to exact matching
        return gt, pred, gt == pred
    
    elif question_type in ['FA', 'FO']:
        def normalize_set(text):
            text = text.strip().lower()
            elements = set()
            
            # 特殊处理嵌套格式 [{'a', 'b'}]
            if text.startswith("[{") and text.endswith("}]"):
                inner_content = text[2:-2].strip()
                for item in re.split(r',\s*', inner_content):
                    clean_item = item.strip().strip("'\"")
                    if clean_item:
                        elements.add(clean_item)
                return elements
            
            # 处理单个集合 {'a', 'b'}
            if text.startswith("{") and text.endswith("}"):
                content = text[1:-1].strip()
                for item in re.split(r',\s*', content):
                    clean_item = item.strip().strip("'\"")
                    if clean_item:
                        elements.add(clean_item)
                return elements
                
            # 处理花括号嵌入在文本中的情况
            set_match = re.search(r'\{([^{}]*)\}', text)
            if set_match:
                content = set_match.group(1).strip()
                for item in re.split(r',\s*', content):
                    clean_item = item.strip()
                    if clean_item:
                        elements.add(clean_item)
                return elements
            
            # 特殊处理 FO 类型问题，提取单字符节点
            if question_type == 'FO':
                # 提取所有单个字母/数字作为节点
                nodes = re.findall(r'\b([a-z]|[0-9])\b', text)
                if nodes:
                    return set(nodes)
                
            # 处理简单的逗号分隔列表
            if ',' in text:
                for item in text.split(','):
                    clean_item = item.strip()
                    if clean_item:
                        # 对于FO类型，尝试提取单字符
                        if question_type == 'FO':
                            match = re.search(r'\b([a-z]|[0-9])\b', clean_item)
                            if match:
                                elements.add(match.group(1))
                        else:
                            elements.add(clean_item)
                return elements

            # 单个元素的情况
            if question_type == 'FO':
                match = re.search(r'\b([a-z]|[0-9])\b', text)
                if match:
                    return {match.group(1)}
            
            return {text.strip()}
                
        # 规范化答案
        gt_processed = normalize_set(gt)
        pred_processed = normalize_set(pred)
        return gt_processed, pred_processed, gt_processed == pred_processed
    return gt, pred, gt == pred

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

        gt, pred, is_match = match_answers(gt_answer, answer_text, gt_question_type)
        print(f"  Expected: {gt}")
        print(f"  Predicted: {pred}")
       
        if is_match:
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
    # YN
    assert match_answers('yes', 'yes, c is a node in the graph. there is an edge c->o, which means there is a directed edge from c to o', 'YN')[2]
    assert match_answers('no', 'no, x->h is not an edge of this graph', 'YN')[2]

    # HM
    assert match_answers('7', 'the graph has 7 nodes', 'HM')[2]

    # MC
    assert match_answers('a', 'a. the graph has 7 nodes', 'MC')[2]
    
    # FA/FO tests
    assert match_answers("{'r', 'l'}", "the maximal valid backdoor adjustment set for treatment s and outcome g in this admg is {l, r}. this set blocks all backdoor paths from s to g.", 'FA')[2]
    assert match_answers("{'x', 'j', 'r'}", "{j, r, x}", 'FA')[2]
    assert match_answers("[{'a', 'b'}]", "the frontdoor adjustment set for treatment t and outcome p in the given dag is {a, b}. variables a and b are both directed towards p (a->p, b->p) and towards t (a->t, b->t), and they are not descendants of t.", 'FA')[2]
    assert match_answers("{'s'}", "s", 'FA')[2]
    assert match_answers("{'x', 'j', 'r'}", "j, r, x", 'FA')[2]
    assert match_answers("j, n, k, t, w", "one valid topological ordering of the graph is: j, n, k, t, w", 'FO')[2]
    assert match_answers("h, k, d, q, f, u", "one valid topological ordering of the graph is: h, d, k, f, q, u", 'FO')[2]
    