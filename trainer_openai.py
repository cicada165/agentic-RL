"""
Search-R1 Trainer with OpenAI API support
Note: OpenAI models cannot be updated, so this version focuses on generation and evaluation.
For full RL training, you would need a local fine-tunable model.
"""
import re
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
import tiktoken

from searchr1_config import SearchR1Config
from trajectory import Trajectory, TokenStep
from search_engine import SearchEngine
import difflib


class SearchR1TrainerOpenAI:
    """Main trainer class for Search-R1 agentic RL using OpenAI API"""
    
    def __init__(self, config: SearchR1Config):
        """Initialize the trainer with configuration."""
        self.config = config
        
        if not config.use_openai:
            raise ValueError("This trainer requires use_openai=True in config")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model_name = config.openai_model
        
        # Initialize tokenizer for token counting (using tiktoken)
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except:
            # Fallback to cl100k_base (used by gpt-4, gpt-3.5-turbo)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize search engine
        self.search_engine = SearchEngine()
        
        # System prompt for the model
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt"""
        prompt = """你是一个具有搜索能力的推理型助手。在解决问题时,你必须遵循以下逻辑步骤:
1. 在 <think> 和 </think>标签内分析问题,确定是否需要外部知识。
2. 如果涉及具体的学校规章制度时,请务必使用 <search>输入用户问题</search> 标签进行查询搜索。
3. 根据以上内容,在 <answer> 输出结果 </answer> 标签内给出准确、简洁的最终结论。

注意:
- 所有回答必须包含 <think> 和 <answer> 标签。
- 如果使用了搜索,尽量使用 <information> 中的原文内容进行回答。

如果与学校规章制度相关,就需要进行搜索。示例:
问题:大学一学期学费是多少钱?
<think>用户询问学费相关问题,这属于学校行政规定。</think>
<search>大学一学期学费是多少钱?</search>
<information>一学期费用是 5 千元</information>
<answer>一学期费用是 5 千元</answer>

如果<information>标签中的内容为空,就回答未找到相关内容: <information></information><answer>未找到相关内容</answer>

如果与学校规章制度不相关,就不需要搜索。示例:
问题:帮我写一篇以勇敢为主题的 100 字的小故事
<think>思考过程</think>
<answer>直接输出结果</answer>"""
        return prompt
    
    def generate_trajectory(self, query: str, max_tokens: int = None) -> Trajectory:
        """
        Generate a trajectory for a given query using OpenAI API.
        
        Args:
            query: Input query
            max_tokens: Maximum tokens to generate
            
        Returns:
            Trajectory object containing the generation path
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        
        trajectory = Trajectory(query=query)
        
        # Build messages for OpenAI chat format
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        generated_text = ""
        token_steps = []
        full_input_ids = []
        generated_positions = []
        
        # Track if we need to handle search
        search_triggered = False
        search_query = None
        
        try:
            # Make API call - try with logprobs first, fallback if not supported
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    logprobs=True,  # Request log probabilities
                    top_logprobs=1  # Get top 1 logprob per token
                )
            except Exception as e:
                # If logprobs not supported, retry without them
                print(f"  Note: logprobs not available for {self.model_name}, using fallback")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
            
            # Extract generated content
            generated_text = response.choices[0].message.content
            
            # Process logprobs if available
            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                content_tokens = response.choices[0].logprobs.content
                for idx, token_info in enumerate(content_tokens):
                    if token_info.top_logprobs:
                        top_logprob = token_info.top_logprobs[0]
                        try:
                            token_id = self.tokenizer.encode(top_logprob.token)[0] if top_logprob.token else 0
                        except:
                            token_id = 0
                        token_text = top_logprob.token
                        log_prob = top_logprob.logprob
                        
                        token_step = TokenStep(
                            token_id=token_id,
                            token_text=token_text,
                            log_prob=log_prob,
                            position=idx
                        )
                        token_steps.append(token_step)
                        generated_positions.append(idx)
            
            # If logprobs not available, create approximate token steps
            if not token_steps:
                # Tokenize the generated text to create approximate token steps
                tokens = self.tokenizer.encode(generated_text)
                for idx, token_id in enumerate(tokens):
                    token_text = self.tokenizer.decode([token_id])
                    token_step = TokenStep(
                        token_id=token_id,
                        token_text=token_text,
                        log_prob=-1.0,  # Placeholder - not available from API
                        position=idx
                    )
                    token_steps.append(token_step)
                    generated_positions.append(idx)
            
            # Handle search if triggered
            if "<search>" in generated_text:
                search_match = re.search(r'<search>(.*?)</search>', generated_text, re.DOTALL)
                if search_match:
                    search_query = search_match.group(1).strip()
                    search_result = self.search_engine.search(search_query)
                    information_tag = f"<information>{search_result}</information>"
                    
                    # Insert information after search tag
                    generated_text = generated_text.replace(
                        f"<search>{search_query}</search>",
                        f"<search>{search_query}</search>{information_tag}",
                        1
                    )
                    search_triggered = True
            
            # Extract final answer
            answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
            if answer_match:
                trajectory.final_answer = answer_match.group(1).strip()
            
        except Exception as e:
            print(f"Error generating trajectory: {e}")
            generated_text = f"<think>Error occurred</think><answer>Error: {str(e)}</answer>"
        
        trajectory.token_steps = token_steps
        trajectory.generated_text = generated_text
        trajectory.full_input_ids = full_input_ids
        trajectory.generated_positions = generated_positions
        
        return trajectory
    
    def check_format_correctness(self, generated_text: str) -> float:
        """
        Check if the generated text follows the required format.
        
        Returns:
            -1.0 if format is incorrect, 0.5 if correct
        """
        # Remove think tags and their content
        generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
        
        # Check answer tags
        answer_start_count = generated_text.count("<answer>")
        answer_end_count = generated_text.count("</answer>")
        
        # Answer tag must appear exactly once
        if answer_start_count != 1 or answer_end_count != 1:
            return -1.0
        
        # Check if answer is at the end
        answer_start_pos = generated_text.rfind("<answer>")
        answer_end_pos = generated_text.rfind("</answer>")
        
        if (not generated_text.endswith("</answer>")) or answer_start_pos > answer_end_pos:
            return -1.0
        
        # Check search and information tags
        start_search_tag = generated_text.count("<search>")
        end_search_tag = generated_text.count("</search>")
        start_information_tag = generated_text.count("<information>")
        end_information_tag = generated_text.count("</information>")
        
        # Must appear in pairs
        tag_pair_check = (start_search_tag == end_search_tag) and (start_information_tag == end_information_tag)
        
        # Two types of tags must exist/not exist simultaneously
        tag_exist_check = (start_search_tag == start_information_tag)
        
        if not (tag_pair_check and tag_exist_check):
            return -1.0
        
        return 0.5
    
    def check_answer_correctness(self, final_answer: str, ground_truth: str) -> float:
        """
        Check if the answer is correct.
        
        Returns:
            0.0 if incorrect, 1.0 if similar, 2.0 if exact match
        """
        if not final_answer or not ground_truth:
            return 0.0
        
        # Exact match
        if final_answer == ground_truth:
            return 2.0
        
        # Special case: "no content found"
        if final_answer == '未找到相关内容':
            return 0.5
        
        # Preprocess: remove spaces and normalize case
        pred = "".join(final_answer.split()).lower()
        target = "".join(ground_truth.split()).lower()
        
        if not pred:
            return 0.0
        
        # Calculate similarity
        matcher = difflib.SequenceMatcher(None, pred, target)
        similarity = matcher.ratio()
        
        # Set hard threshold
        if similarity < 0.5:
            return 0.0
        
        return 1.0
    
    def compute_reward(self, trajectory: Trajectory, ground_truth: str) -> float:
        """
        Compute reward for a trajectory.
        Result-oriented reward: format check + answer correctness.
        
        Args:
            trajectory: The trajectory to evaluate
            ground_truth: Ground truth answer
            
        Returns:
            Total reward value
        """
        # Format check
        format_reward = self.check_format_correctness(trajectory.generated_text)
        
        # Answer check
        answer_reward = self.check_answer_correctness(trajectory.final_answer, ground_truth)
        
        # Total reward
        total_reward = format_reward + answer_reward
        
        return total_reward
    
    def compute_advantages(self, rewards: List[float]) -> np.ndarray:
        """
        Calculate relative advantages within a group.
        A_i = (r_i - μ) / σ
        
        Args:
            rewards: List of reward values
            
        Returns:
            Array of advantages
        """
        rewards_array = np.array(rewards, dtype=np.float32)
        
        # If only one sample, return 0 (no advantage)
        if len(rewards) == 1:
            return np.zeros_like(rewards_array)
        
        # Calculate mean and standard deviation
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array, ddof=0) + 1e-8  # Use population std
        
        # Calculate advantages
        advantages = (rewards_array - mean_reward) / std_reward
        
        return advantages
    
    def train_step(self, queries: List[str], ground_truths: List[str], group_size: int = None) -> Dict[str, float]:
        """
        Execute one training step - evaluation and advantage calculation.
        Note: OpenAI models cannot be updated, so this only computes rewards and advantages.
        
        Args:
            queries: List of input queries
            ground_truths: List of ground truth answers
            group_size: Number of trajectories to generate per query
            
        Returns:
            Dictionary of metrics
        """
        if group_size is None:
            group_size = self.config.group_size
        
        all_trajectories = []
        
        for query, truth in zip(queries, ground_truths):
            group_trajectories = []
            
            # 1. Generate multiple trajectories for the same query (Group Sampling)
            for _ in range(group_size):
                trajectory = self.generate_trajectory(query, max_tokens=self.config.max_tokens)
                trajectory.reward = self.compute_reward(trajectory, truth)
                group_trajectories.append(trajectory)
            
            # 2. Calculate group-internal reward statistics
            group_rewards = np.array([t.reward for t in group_trajectories], dtype=np.float32)
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards, ddof=0) + 1e-8
            
            # 3. Calculate relative advantage and store in trajectory object
            for t in group_trajectories:
                # Normalization: (current reward - group average) / group standard deviation
                t.advantage = (t.reward - mean_reward) / std_reward
                all_trajectories.append(t)
        
        # Calculate statistics
        rewards = [traj.reward for traj in all_trajectories]
        avg_tokens = np.mean([len(traj.token_steps) for traj in all_trajectories])
        search_count = sum(1 for traj in all_trajectories if "<search>" in traj.generated_text)
        
        metrics = {
            "avg_reward": np.mean(rewards),
            "avg_tokens": avg_tokens,
            "search_trajectories": search_count / len(all_trajectories) if all_trajectories else 0.0,
            "trajectories": all_trajectories,
            "note": "OpenAI models cannot be updated - this is evaluation only"
        }
        
        return metrics
