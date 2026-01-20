"""
Search-R1 Trainer implementing GRPO algorithm
"""
import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from searchr1_config import SearchR1Config
from trajectory import Trajectory, TokenStep
from search_engine import SearchEngine
import difflib


class SearchR1Trainer:
    """Main trainer class for Search-R1 agentic RL"""
    
    def __init__(self, config: SearchR1Config):
        """Initialize the trainer with configuration."""
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading model from {config.model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        if self.device.type == "cpu":
            self.model = self.model.to(self.device)
        
        # Initialize search engine
        self.search_engine = SearchEngine()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # System prompt for the model
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """Create the system prompt template"""
        prompt = '''<|im_start|>system
你是一个具有搜索能力的推理型助手。在解决问题时,你必须遵循以下逻辑步骤:
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
<answer>一学期费用是 5 千元</answer>(如果有<search>, 就根据<information>中的内容回答)

如果<information>标签中的内容为空,就回答未找到相关内容: <information></information><answer>未找到相关内容</answer>

如果与学校规章制度不相关,就不需要搜索。示例:
问题:帮我写一篇以勇敢为主题的 100 字的小故事
<think>思考过程</think>
<answer>直接输出结果</answer>
<|im_end|>
<|im_start|>user
{query}<|im_end|>
'''
        return prompt
    
    def generate_trajectory(self, query: str, max_tokens: int = None) -> Trajectory:
        """
        Generate a trajectory for a given query.
        
        Args:
            query: Input query
            max_tokens: Maximum tokens to generate
            
        Returns:
            Trajectory object containing the generation path
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        
        trajectory = Trajectory(query=query)
        token_steps = []
        
        # Format prompt
        prompt = self.prompt_template.format(query=query)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Track full input sequence
        full_input_ids = input_ids[0].tolist()
        generated_positions = []
        
        self.model.eval()
        generated_text = ""
        current_input = input_ids
        
        with torch.no_grad():
            for step in range(max_tokens):
                # Generate next token
                outputs = self.model(current_input)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Sample token
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                token_text = self.tokenizer.decode([token_id])
                log_prob = log_probs[0, token_id].item()
                
                # Record token step
                token_step = TokenStep(
                    token_id=token_id,
                    token_text=token_text,
                    log_prob=log_prob,
                    position=len(full_input_ids) + step
                )
                token_steps.append(token_step)
                
                # Update generated text
                generated_text += token_text
                generated_positions.append(len(full_input_ids) + step)
                
                # Check for search trigger
                if "<search>" in generated_text:
                    # Extract search query
                    search_match = re.search(r'<search>(.*?)</search>', generated_text, re.DOTALL)
                    if search_match:
                        search_query = search_match.group(1).strip()
                        # Perform search
                        search_result = self.search_engine.search(search_query)
                        # Insert information tag
                        information_tag = f"<information>{search_result}</information>"
                        generated_text = generated_text.replace(
                            f"<search>{search_query}</search>",
                            f"<search>{search_query}</search>{information_tag}"
                        )
                        # Update input with search result
                        info_ids = self.tokenizer.encode(information_tag, return_tensors="pt").to(self.device)
                        current_input = torch.cat([current_input, info_ids], dim=1)
                        full_input_ids.extend(info_ids[0].tolist())
                
                # Append token to current input
                token_tensor = torch.tensor([[token_id]]).to(self.device)
                current_input = torch.cat([current_input, token_tensor], dim=1)
                full_input_ids.append(token_id)
                
                # Check for end of answer
                if "</answer>" in generated_text:
                    break
        
        # Extract final answer
        answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
        if answer_match:
            trajectory.final_answer = answer_match.group(1).strip()
        
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
    
    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        Calculate relative advantages within a group.
        A_i = (r_i - μ) / σ
        
        Args:
            rewards: List of reward values
            
        Returns:
            Tensor of advantages
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # If only one sample, return 0 (no advantage)
        if len(rewards) == 1:
            return torch.zeros_like(rewards_tensor)
        
        # Calculate mean and standard deviation
        mean_reward = torch.mean(rewards_tensor)
        std_reward = torch.std(rewards_tensor, unbiased=False) + 1e-8  # Use population std
        
        # Calculate advantages
        advantages = (rewards_tensor - mean_reward) / std_reward
        
        return advantages
    
    def recompute_log_probs(self, trajectories: List[Trajectory]) -> List[torch.Tensor]:
        """
        Recompute log probabilities using current model parameters.
        
        Args:
            trajectories: List of trajectories
            
        Returns:
            List of log probability tensors
        """
        new_log_probs_list = []
        
        self.model.eval()
        with torch.no_grad():
            for traj in trajectories:
                # Reconstruct input
                prompt = self.prompt_template.format(query=traj.query)
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                # Process generated tokens
                log_probs = []
                current_input = input_ids
                
                for token_step in traj.token_steps:
                    outputs = self.model(current_input)
                    logits = outputs.logits[:, -1, :]
                    log_probs_tensor = F.log_softmax(logits, dim=-1)
                    log_prob = log_probs_tensor[0, token_step.token_id].item()
                    log_probs.append(log_prob)
                    
                    # Append token for next step
                    token_tensor = torch.tensor([[token_step.token_id]]).to(self.device)
                    current_input = torch.cat([current_input, token_tensor], dim=1)
                
                new_log_probs_list.append(torch.tensor(log_probs, dtype=torch.float32).to(self.device))
        
        return new_log_probs_list
    
    def compute_kl_divergence(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate K3 approximation of KL divergence required by GRPO.
        K3 = (π_old/π_new) - log(π_old/π_new) - 1
        
        Args:
            old_log_probs: Log probabilities from old policy
            new_log_probs: Log probabilities from new policy
            
        Returns:
            KL divergence (token-level average)
        """
        # Step 1: Calculate π_old/π_new = exp(old_log_probs - new_log_probs)
        ratio_old_new = torch.exp(old_log_probs - new_log_probs)
        
        # Step 2: Calculate log(π_old/π_new) = old_log_probs - new_log_probs
        log_ratio_old_new = old_log_probs - new_log_probs
        
        # Step 3: K3 formula
        k3 = ratio_old_new - log_ratio_old_new - 1
        
        # Take mean (token-level average)
        return torch.mean(k3)
    
    def update_policy(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Update policy using GRPO algorithm.
        
        Args:
            trajectories: List of trajectories with advantages
            
        Returns:
            Dictionary of training metrics
        """
        if not trajectories:
            return {"loss": 0.0, "kl_div": 0.0}
        
        self.model.train()
        
        # Calculate rewards and advantages
        rewards = [traj.reward for traj in trajectories]
        advantages = self.compute_advantages(rewards)
        
        # Extract old log probabilities
        old_log_probs_list = []
        for traj in trajectories:
            old_probs = torch.tensor([step.log_prob for step in traj.token_steps]).to(self.device)
            old_log_probs_list.append(old_probs)
        
        # Update model multiple times with same batch
        update_times = self.config.update_times
        
        for _ in range(update_times):
            # Recompute new log probabilities
            new_log_probs_list = self.recompute_log_probs(trajectories)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Collect all samples' loss
            all_policy_losses = []
            all_kl_divs = []
            
            for i, traj in enumerate(trajectories):
                new_log_probs = new_log_probs_list[i]
                old_log_probs = old_log_probs_list[i]
                
                if len(old_log_probs) != len(new_log_probs):
                    continue
                
                # Calculate probability ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Extend advantage to all tokens
                traj_advantage = advantages[i].repeat(len(ratio)).to(self.device)
                
                # PPO clipped objective
                surr1 = ratio * traj_advantage
                surr2 = (
                    torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                    * traj_advantage
                )
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # KL divergence
                kl_div = self.compute_kl_divergence(old_log_probs, new_log_probs)
                
                all_policy_losses.append(policy_loss)
                all_kl_divs.append(kl_div)
            
            # Calculate total loss
            if all_policy_losses:
                total_policy_loss = torch.stack(all_policy_losses).mean()
                total_kl_div = torch.stack(all_kl_divs).mean()
                total_loss = total_policy_loss + self.config.beta * total_kl_div
                
                # Backpropagate
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Update parameters
                self.optimizer.step()
                
                # Record statistics
                avg_loss = total_loss.item()
                avg_kl = total_kl_div.item()
            else:
                avg_loss = 0.0
                avg_kl = 0.0
        
        # Explicitly clear cache
        torch.cuda.empty_cache()
        
        return {
            "loss": avg_loss,
            "kl_div": avg_kl,
            "avg_reward": np.mean(rewards),
            "beta": self.config.beta
        }
    
    def train_step(self, queries: List[str], ground_truths: List[str], group_size: int = None) -> Dict[str, float]:
        """
        Execute one training step - adapting GRPO group-internal advantage calculation.
        
        Args:
            queries: List of input queries
            ground_truths: List of ground truth answers
            group_size: Number of trajectories to generate per query
            
        Returns:
            Dictionary of training metrics
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
            group_rewards = torch.tensor([t.reward for t in group_trajectories], dtype=torch.float32)
            mean_reward = group_rewards.mean()
            std_reward = group_rewards.std() + 1e-8
            
            # 3. Calculate relative advantage and store in trajectory object
            for t in group_trajectories:
                # Normalization: (current reward - group average) / group standard deviation
                t.advantage = (t.reward - mean_reward.item()) / std_reward.item()
                all_trajectories.append(t)
        
        # 4. Call update policy (note: update_policy internally reads from traj.advantage)
        metrics = self.update_policy(all_trajectories)
        
        # Calculate average token count
        avg_tokens = np.mean([len(traj.token_steps) for traj in all_trajectories])
        
        # Calculate number of trajectories containing search instructions
        search_count = sum(1 for traj in all_trajectories if "<search>" in traj.generated_text)
        
        # Update metrics
        metrics.update({
            "avg_tokens": avg_tokens,
            "search_trajectories": search_count / len(all_trajectories) if all_trajectories else 0.0,
            "trajectories": all_trajectories
        })
        
        torch.cuda.empty_cache()
        
        return metrics
