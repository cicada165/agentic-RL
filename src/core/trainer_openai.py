"""
Search-R1 Trainer with OpenAI/GitHub API support.
"""
import re
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
import tiktoken

from src.utils.config import SearchR1Config
from src.core.trajectory import Trajectory, TokenStep
from src.agent.search_engine import SearchEngine
import difflib


class SearchR1TrainerOpenAI:
    """Main trainer class for Search-R1 agentic RL using OpenAI/GitHub API"""
    
    def __init__(self, config: SearchR1Config):
        """Initialize the trainer with configuration."""
        self.config = config
        
        if not config.use_openai:
            raise ValueError("This trainer requires use_openai=True in config")
        
        # Initialize OpenAI client
        # Handle GitHub Models specific configuration
        api_key = config.openai_api_key
        base_url = config.openai_api_base
        
        print(f"Initializing API client with base_url={base_url}")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = config.openai_model
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o") # Default to gpt-4o encoding
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.search_engine = SearchEngine()
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt"""
        return """你是一个具有搜索能力的推理型助手。在解决问题时,你必须遵循以下逻辑步骤:
1. 在 <think> 和 </think>标签内分析问题,确定是否需要外部知识。
2. 如果涉及具体的学校规章制度时,请务必使用 <search>输入用户问题</search> 标签进行查询搜索。
3. 根据以上内容,在 <answer> 输出结果 </answer> 标签内给出准确、简洁的最终结论。

注意:
- 所有回答必须包含 <think> 和 <answer> 标签。
- 如果使用了搜索,尽量使用 <information> 中的原文内容进行回答。

示例:
问题:大学一学期学费是多少钱?
<think>用户询问学费相关问题,这属于学校行政规定。</think>
<search>大学一学期学费是多少钱?</search>
<information>一学期费用是 5 千元</information>
<answer>一学期费用是 5 千元</answer>

如果<information>标签中的内容为空,就回答未找到相关内容: <information></information><answer>未找到相关内容</answer>
"""
    
    def generate_trajectory(self, query: str, max_tokens: int = None) -> Trajectory:
        """Generate a trajectory using API."""
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        
        trajectory = Trajectory(query=query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        generated_text = ""
        token_steps = []
        full_input_ids = []  # Not fully trackable with API
        
        try:
            # First generation attempt
            response = self._call_api(messages, max_tokens)
            generated_text = response.choices[0].message.content
            
            # Check for search
            if "<search>" in generated_text:
                search_match = re.search(r'<search>(.*?)</search>', generated_text, re.DOTALL)
                if search_match:
                    search_query = search_match.group(1).strip()
                    search_result = self.search_engine.search(search_query)
                    
                    # We need to simulate the multi-step process
                    # 1. Truncate generated text up to </search>
                    # 2. Append <information>...</information>
                    # 3. Feed back to model to complete prediction
                    
                    search_end_idx = generated_text.find("</search>") + len("</search>")
                    pre_search_text = generated_text[:search_end_idx]
                    
                    information_block = f"<information>{search_result}</information>"
                    
                    # Update messages
                    messages.append({"role": "assistant", "content": pre_search_text + information_block})
                    
                    # Continue generation
                    # Note: We are cheating a bit here by appending to assistant output.
                    # Ideally we should stop generation at </search>, but API doesn't support regex stop easily.
                    # GitHub Models might not support 'stop' sequences universally or complex partial completion.
                    # Best approach: Let it generate, if we see search, we perform it, append info, and ask for completion?
                    # The current approach in local trainer is:
                    #  - Generates token by token.
                    #  - If <search> found, pause, insert info, continue.
                    
                    # For API:
                    # We can't do token-by-token control easily.
                    # Simplified: If search is found, we just replace it in the text context for now?
                    # No, we want the MODEL to use the info.
                    
                    # Re-prompting strategy:
                    # 1. Assistant: <think>...<search>query</search>
                    # 2. User: <information>result</information>
                    # 3. Assistant: <answer>...
                    
                    # Let's use a simplified approach for evaluation:
                    # If we see <search>, we assume the model *wanted* to search.
                    # We construct a new message history where we *inject* the info.
                    
                    new_messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query},
                        # We need to split the assistant's previous output?
                        # It's hard to split perfectly if it generated past the search.
                        # Let's take the text up to </search>
                    ]
                     
                    # If the model already generated the answer *despite* searching (hallucinating info?), we might keep it.
                    # But if we want to test "Agentic" flow:
                    
                    # Let's just do single turn for now as per original script logic, 
                    # OR improve it slightly by appending info if the model stopped or we can prompt continuation.
                    
                    # Original script logic just replaced text. That's purely cosmetic for the reward, 
                    # it doesn't help the model *use* the info.
                    # Let's try to be a bit smarter.
                    
                    if "<answer>" not in generated_text:
                         # Model stopped or needs info
                         messages.append({"role": "assistant", "content": pre_search_text + information_block})
                         resp2 = self._call_api(messages, max_tokens)
                         generated_text = pre_search_text + information_block + resp2.choices[0].message.content
                    else:
                        # Model already answered. We just insert the info tag for format correctness?
                        # Or checking if it hallucinated.
                        generated_text = generated_text.replace(
                            f"<search>{search_query}</search>",
                            f"<search>{search_query}</search>{information_block}"
                        )

        except Exception as e:
            print(f"API Error: {e}")
            generated_text = f"Error: {str(e)}"

        trajectory.generated_text = generated_text
        
        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
        if answer_match:
            trajectory.final_answer = answer_match.group(1).strip()
            
        # No token steps for API to save complexity/API costs/rate limits
        
        return trajectory

    def _call_api(self, messages, max_tokens):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )

    # Reuse reward functions from src.core? 
    # To keep this standalone-ish or use shared code?
    # Let's use shared logic
    
    def compute_reward(self, trajectory: Trajectory, ground_truth: str) -> float:
        from src.rewards.reward import compute_reward
        return compute_reward(trajectory, ground_truth)

    def train_step(self, queries: List[str], ground_truths: List[str]) -> Dict[str, float]:
        all_trajectories = []
        for query, truth in zip(queries, ground_truths):
            # No group sampling, just 1 per query for eval to save tokens
            traj = self.generate_trajectory(query)
            traj.reward = self.compute_reward(traj, truth)
            all_trajectories.append(traj)
            
        return {
            "avg_reward": np.mean([t.reward for t in all_trajectories]),
            "trajectories": all_trajectories
        }
