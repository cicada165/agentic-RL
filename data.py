"""
Dataset creation for Search-R1 training
Using school regulation data from RAG project
"""
from typing import List, Tuple


def create_training_data() -> Tuple[List[str], List[str]]:
    """
    Create training data with queries and ground truths.
    
    Returns:
        Tuple of (queries, ground_truths) lists
    """
    queries = [
        "学校对入学新生的复查应在多长时间内完成?",
        "留校察看处分的期限一般是多久?",
        # "高校学生在校园内禁止从事哪些行为?",
        # "学校对校园内的宗教活动有什么规定?"
    ]
    
    ground_truths = [
        "学生入学后,学校应当在3个月内按国家招生规定复查,内容包括录取手续、资格真实性、身份一致性、身心健康状况等。",
        "除开除学籍外,警告、严重警告、记过、留校察看处分一般设置6-12个月期限,到期按学校规定程序解除,解除后不影响后续权益。",
        # "学生不得酗酒、打架斗殴、赌博、吸毒,不得传播非法书刊音像制品,不得参与非法传销、邪教/封建迷信活动,不得从事有损大学生形象的活动。",
        # "学校坚持教育与宗教相分离原则,任何组织和个人不得在学校进行宗教活动。"
    ]
    
    return queries, ground_truths
