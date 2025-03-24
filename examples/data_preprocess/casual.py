""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import random

def make_prefix(dp, template_type):
    # quiz = dp['quiz']
    quiz = dp['description'] + '\n' + dp['question']
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a casual graph reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags. \n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/casual')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='./data/casual/raw/clear')
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'clear'

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # 如果 answer 是列表，将其转换为字符串
                        if 'answer' in data and isinstance(data['answer'], list):
                            data['answer'] = json.dumps(data['answer'])
                        yield data
        else:
            for filename in os.listdir(path):
                if filename.endswith('.json') or filename.endswith('.jsonl'):
                    file_path = os.path.join(path, filename)
                    print(f"Processing file: {file_path}")
                    with open(file_path) as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                # 如果 answer 是列表，将其转换为字符串
                                if 'answer' in data and isinstance(data['answer'], list):
                                    data['answer'] = json.dumps(data['answer'])
                                yield data

    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(f"Total dataset size: {len(raw_dataset)}")

    # 按照 task_type 对数据集进行分组
    task_types = {}
    for idx, item in enumerate(raw_dataset):
        task_type = item.get('task_type', 'default')  # 如果没有 task_type 则使用 'default'
        if task_type not in task_types:
            task_types[task_type] = []
        task_types[task_type].append(idx)

    # 对每种 task_type 进行 80-20 分割
    train_indices = []
    test_indices = []
    
    for task_type, indices in task_types.items():
        n_samples = len(indices)
        n_train = int(n_samples * 0.8)
        
        # 打乱该 task_type 的索引
        random.shuffle(indices)
        
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
        
        print(f"Task type '{task_type}': total {n_samples}, train {n_train}, test {len(indices) - n_train}")

    # 使用索引创建训练集和测试集
    train_dataset = raw_dataset.select(train_indices)
    test_dataset = raw_dataset.select(test_indices)

    print(f"Final split: train {len(train_dataset)}, test {len(test_dataset)}")

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "casual",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example['answer']
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'task_type': example['task_type'],
                    'uid': example['uid'],
                    'sid': example['sid'],
                    'question_type': example['question_type'],
                    'graph_type': example['graph_type']
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)