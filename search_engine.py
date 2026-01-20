"""
Search Engine for Search-R1
In practical scenarios, this can be changed to web search or RAG search
"""
from typing import Dict


class SearchEngine:
    """Simple keyword-based search engine. Can be extended to web search or RAG."""
    
    def __init__(self):
        """Initialize the search engine with a knowledge base."""
        self.knowledge_base: Dict[str, str] = {
            "复查": "学生入学后,学校应当在3个月内按国家招生规定复查,内容包括录取手续、资格真实性、身份一致性、身心健康状况等。",
            "处分": "除开除学籍外,警告、严重警告、记过、留校察看处分一般设置6-12个月期限,到期按学校规定程序解除,解除后不影响后续权益。",
            "禁止行为": "学生不得酗酒、打架斗殴、赌博、吸毒,不得传播非法书刊音像制品,不得参与非法传销、邪教/封建迷信活动,不得从事有损大学生形象的活动。",
            "宗教": "学校坚持教育与宗教相分离原则,任何组织和个人不得在学校进行宗教活动。"
        }
    
    def search(self, query: str) -> str:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query string
            
        Returns:
            Retrieved information or a message indicating no information found
        """
        query_lower = query.lower().strip()
        
        # Simple keyword matching
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                return value
        
        return f"No information found for: {query}"
