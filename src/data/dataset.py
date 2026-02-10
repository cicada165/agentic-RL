"""
Dataset creation for Search-R1.
"""
import json
import os
from typing import Tuple, List, Optional

def create_training_data() -> Tuple[List[str], List[str]]:
    """
    Create training data pairs using default hardcoded data.
    """
    queries = [
        "学校对入学新生的复查应在多长时间内完成?",
        "留校察看处分的期限一般是多久?",
        "学生在校期间可以进行宗教活动吗?",
        "参与非法传销会受什么处分?"
    ]
    ground_truths = [
        "学生入学后,学校应当在3个月内按国家招生规定复查,内容包括录取手续、资格真实性、身份一致性、身心健康状况等。",
        "除开除学籍外,警告、严重警告、记过、留校察看处分一般设置6-12个月期限,到期按学校规定程序解除,解除后不影响后续权益。",
        "学校坚持教育与宗教相分离原则,任何组织和个人不得在学校进行宗教活动。",
        "学生不得参与非法传销、邪教/封建迷信活动,不得从事有损大学生形象的活动。"
    ]
    return queries, ground_truths

def load_from_json(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load query-answer pairs from a JSON file.
    
    Format:
    [
        {"query": "...", "answer": "..."},
        ...
    ]
    or
    {
        "queries": ["...", ...],
        "ground_truths": ["...", ...]
    }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    queries = []
    ground_truths = []
    
    if isinstance(data, list):
        for item in data:
            if "query" in item and "answer" in item:
                queries.append(item["query"])
                ground_truths.append(item["answer"])
            elif "question" in item and "answer" in item: # Common variation
                queries.append(item["question"])
                ground_truths.append(item["answer"])
    elif isinstance(data, dict):
        if "queries" in data and "ground_truths" in data:
            queries = data["queries"]
            ground_truths = data["ground_truths"]
        elif "questions" in data and "answers" in data:
            queries = data["questions"]
            ground_truths = data["answers"]
            
    if not queries:
        raise ValueError("No valid data found in JSON file")
        
    return queries, ground_truths
