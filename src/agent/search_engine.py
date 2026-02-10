"""
Search Engine interface and implementation for Search-R1.
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

class SearchEngine(ABC):
    """Abstract interface for search functionality."""
    
    @abstractmethod
    def search(self, query: str) -> str:
        """
        Search for information given a query string.
        """
        pass


class MockSearchEngine(SearchEngine):
    """
    Mock search engine with a static knowledge base for testing.
    """
    
    def __init__(self):
        self.knowledge_base: Dict[str, str] = {
            "复查": "学生入学后,学校应当在3个月内按国家招生规定复查,内容包括录取手续、资格真实性、身份一致性、身心健康状况等。",
            "处分": "除开除学籍外,警告、严重警告、记过、留校察看处分一般设置6-12个月期限,到期按学校规定程序解除,解除后不影响后续权益。",
            "禁止行为": "学生不得酗酒、打架斗殴、赌博、吸毒,不得传播非法书刊音像制品,不得参与非法传销、邪教/封建迷信活动,不得从事有损大学生形象的活动。",
            "宗教": "学校坚持教育与宗教相分离原则,任何组织和个人不得在学校进行宗教活动。"
        }
    
    def search(self, query: str) -> str:
        query_lower = query.lower().strip()
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                return value
        return f"No information found for: {query}"


class DuckDuckGoSearchEngine(SearchEngine):
    """
    Real web search using DuckDuckGo (free, no API key required).
    Requires: pip install duckduckgo-search
    """
    
    def __init__(self, max_results: int = 3):
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.max_results = max_results
            self.available = True
        except ImportError:
            print("Warning: duckduckgo-search not installed. DuckDuckGoSearchEngine disabled.")
            self.available = False

    def search(self, query: str) -> str:
        if not self.available:
            return "Search engine not available (duckduckgo-search missing)."
            
        try:
            results = self.ddgs.text(query, max_results=self.max_results)
            if not results:
                return f"No information found for: {query}"
            
            # Combine results
            combined_text = ""
            for i, res in enumerate(results):
                combined_text += f"Result {i+1}: {res['body']}\n"
            
            return combined_text.strip()
            
        except Exception as e:
            return f"Search error: {str(e)}"


class GoogleSearchEngine(SearchEngine):
    """
    Real web search using Google Custom Search API.
    Requires: 
    - pip install google-api-python-client
    - GOOGLE_API_KEY environment variable
    - GOOGLE_CSE_ID environment variable (Custom Search Engine ID)
    """
    
    def __init__(self, api_key: Optional[str] = None, cse_id: Optional[str] = None, max_results: int = 3):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")
        self.max_results = max_results
        self.available = False
        
        if self.api_key and self.cse_id:
            try:
                from googleapiclient.discovery import build
                self.service = build("customsearch", "v1", developerKey=self.api_key)
                self.available = True
            except ImportError:
                print("Warning: google-api-python-client not installed.")
            except Exception as e:
                print(f"Warning: Failed to initialize Google Search: {e}")
        else:
            print("Warning: Google API Key or CSE ID missing.")

    def search(self, query: str) -> str:
        if not self.available:
            return "Google Search not configured or unavailable."
            
        try:
            res = self.service.cse().list(q=query, cx=self.cse_id, num=self.max_results).execute()
            items = res.get('items', [])
            
            if not items:
                return f"No information found for: {query}"
                
            combined_text = ""
            for i, item in enumerate(items):
                snippet = item.get('snippet', '')
                combined_text += f"Result {i+1}: {snippet}\n"
                
            return combined_text.strip()
            
        except Exception as e:
            return f"Google Search error: {str(e)}"
