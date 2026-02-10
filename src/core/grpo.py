"""
GROP (Group Relative Policy Optimization) Trainer for Search-R1.
This module implements the core training loop, trajectory generation, and policy update logic.
"""

import re
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

from src.utils.config import SearchR1Config
from src.core.trajectory import Trajectory, TokenStep
from src.agent.search_engine import SearchEngine, MockSearchEngine
from src.agent.model import ModelFactory
from src.rewards.reward import compute_reward

class SearchR1Trainer:
    """Main trainer class for Search-R1 agentic RL using GRPO."""
    
    def __init__(self, config: SearchR1Config):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: SearchR1Config object with hyperparameters.
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model, self.tokenizer = ModelFactory.load_model_and_tokenizer(
            config.model_name_or_path, 
            config.device
        )
        
        # Initialize search engine (Using Mock for now, can be swapped)
        self.search_engine = MockSearchEngine()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # System prompt
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """Create the system prompt template."""
        return '''<|im_start|>system
你是一个具有搜索能力的推理型助手。在解决问题时,你必须遵循以下逻辑步骤:
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

<|im_end|>
<|im_start|>user
{query}<|im_end|>
'''
    
    def generate_trajectory(self, query: str, max_tokens: int = None) -> Trajectory:
        """
        Generate a trajectory for a given query with search capability.
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
                # Generete next token
                outputs = self.model(current_input)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Sample token
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                token_text = self.tokenizer.decode([token_id])
                log_prob = log_probs[0, token_id].item()
                
                # Record step
                position = len(full_input_ids)
                token_step = TokenStep(
                    token_id=token_id,
                    token_text=token_text,
                    log_prob=log_prob,
                    position=position
                )
                trajectory.add_step(token_step)
                
                # Update text and processing
                generated_text += token_text
                generated_positions.append(position)
                
                # Check for search
                if "<search>" in generated_text and "</search>" in generated_text:
                    # Check if we just closed a search tag that hasn't been processed
                    # This logic assumes we process immediately after </search> is generated
                    # A more robust way is to check if we have a search tag but no info tag
                     if "<search>" in generated_text and "<information>" not in generated_text:
                        search_match = re.search(r'<search>(.*?)</search>', generated_text, re.DOTALL)
                        if search_match:
                            search_query = search_match.group(1).strip()
                            search_result = self.search_engine.search(search_query)
                            
                            information_block = f"<information>{search_result}</information>"
                            
                            # We need to insert this into the generation context
                            # AND update the generated text in the trajectory? 
                            # Actually, for the model to see it, we must append it to current_input
                            
                            # Note: The 'generated_text' so far contains <search>...</search>
                            # We append <information>...</information> to it.
                            generated_text += information_block
                            
                            # Tokenize info block
                            info_ids = self.tokenizer.encode(information_block, return_tensors="pt").to(self.device)
                            current_input = torch.cat([current_input, info_ids], dim=1)
                            full_input_ids.extend(info_ids[0].tolist())
                            
                            # IMPORTANT: We do NOT create TokenSteps for inserted text
                            # because the model didn't generate them (it's environment feedback)
                            # However, 'generated_positions' indices for subsequent tokens must shift?
                            # The current implementation uses simple append, so next token will have 
                            # position = len(full_input_ids) + step. 
                            # Since full_input_ids grew by info_ids length, the next position will jump correctly.
                
                # Update input
                token_tensor = torch.tensor([[token_id]]).to(self.device)
                current_input = torch.cat([current_input, token_tensor], dim=1)
                full_input_ids.append(token_id)
                
                if "</answer>" in generated_text:
                    break
        
        # Extract final answer
        answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
        if answer_match:
            trajectory.final_answer = answer_match.group(1).strip()
            
        trajectory.generated_text = generated_text
        trajectory.full_input_ids = full_input_ids
        trajectory.generated_positions = generated_positions
        
        return trajectory

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        Calculate group-relative advantages.
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        if len(rewards) == 1:
            return torch.zeros_like(rewards_tensor)
            
        mean_reward = torch.mean(rewards_tensor)
        std_reward = torch.std(rewards_tensor, unbiased=False) + 1e-8
        
        return (rewards_tensor - mean_reward) / std_reward

    def compute_kl_divergence(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate K3 approximation of KL divergence.
        """
        ratio = torch.exp(old_log_probs - new_log_probs)
        log_ratio = old_log_probs - new_log_probs
        k3 = ratio - log_ratio - 1
        return torch.mean(k3)

    def recompute_log_probs(self, trajectories: List[Trajectory]) -> List[torch.Tensor]:
        """
        Recompute log probabilities using current model parameters.
        """
        new_log_probs_list = []
        # self.model.eval() # We should probably be in train mode? Or eval?
        # Standard PPO uses eval for rollout, but for update we need gradients.
        # But here we are just computing log probs. 
        # The model is already in train mode in update_policy.
        
        for traj in trajectories:
            prompt = self.prompt_template.format(query=traj.query)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            current_input = input_ids
            traj_log_probs = []
            
            # ... reconstruction ...
            
            full_seq = torch.tensor([traj.full_input_ids]).to(self.device)
            outputs = self.model(full_seq)
            # logits shape: [1, seq_len, vocab_size]
            # Logits at index i predicts token at i+1
            
            all_logits = outputs.logits[0] # [seq_len, vocab]
            all_log_probs = F.log_softmax(all_logits, dim=-1)
            
            # For each generated token at position p, its log_prob is at all_log_probs[p-1]
            # selecting the token_id of the token at p.
            
            step_log_probs = []
            for step in traj.token_steps:
                # position includes prompt len.
                # e.g. prompt len 10. generated first token at 10 (0-indexed).
                # input_ids has 11 elements. 
                # logits[9] predicts token at 10.
                pos = step.position
                token_id = step.token_id
                
                # Logic check:
                # input: [A, B, C]
                # logits: [pred_B, pred_C, pred_D]
                # if we generated D at pos 3. 
                # we want log_prob of D given [A, B, C].
                # This is at logits index 2 (len-1).
                
                if pos > 0:
                    lp = all_log_probs[pos-1, token_id] # Keep gradient!
                    step_log_probs.append(lp)
            
            new_log_probs_list.append(torch.stack(step_log_probs).to(self.device))
                
        return new_log_probs_list

    def update_policy(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Update policy using GRPO.
        """
        if not trajectories:
            return {"loss": 0.0, "kl_div": 0.0}
            
        self.model.train()
        rewards = [traj.reward for traj in trajectories]
        advantages = self.compute_advantages(rewards)
        
        old_log_probs_list = [
            torch.tensor([s.log_prob for s in t.token_steps]).to(self.device)
            for t in trajectories
        ]
        
        avg_loss = 0.0
        avg_kl = 0.0
        
        for _ in range(self.config.update_times):
            new_log_probs_list = self.recompute_log_probs(trajectories)
            
            self.optimizer.zero_grad()
            all_policy_losses = []
            all_kl_divs = []
            
            for i, traj in enumerate(trajectories):
                new_logs = new_log_probs_list[i]
                old_logs = old_log_probs_list[i]
                
                if len(new_logs) != len(old_logs):
                    # Length mismatch might happen if logic differs, skip
                    continue
                    
                ratio = torch.exp(new_logs - old_logs)
                adv = advantages[i].repeat(len(ratio)).to(self.device)
                
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * adv
                
                policy_loss = -torch.min(surr1, surr2).mean()
                kl_div = self.compute_kl_divergence(old_logs, new_logs)
                
                all_policy_losses.append(policy_loss)
                all_kl_divs.append(kl_div)
            
            if all_policy_losses:
                total_policy_loss = torch.stack(all_policy_losses).mean()
                total_kl = torch.stack(all_kl_divs).mean()
                total_loss = total_policy_loss + self.config.beta * total_kl
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                avg_loss = total_loss.item()
                avg_kl = total_kl.item()
                
        torch.cuda.empty_cache()
        return {
            "loss": avg_loss, 
            "kl_div": avg_kl, 
            "avg_reward": np.mean(rewards),
            "beta": self.config.beta
        }

    def save_checkpoint(self, path: str, epoch: int, step: int):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'config': self.config
        }, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str) -> Tuple[int, int]:
        """
        Load training checkpoint.
        Returns (epoch, step) to resume from.
        """
        if not os.path.exists(path):
            print(f"Checkpoint {path} not found.")
            return 0, 0
            
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path} (Epoch {checkpoint['epoch']}, Step {checkpoint['step']})")
        return checkpoint['epoch'], checkpoint['step']

    def train_step(self, queries: List[str], ground_truths: List[str], accumulation_steps: int = 1) -> Dict[str, float]:
        """
        Execute one training step with optional gradient accumulation.
        """
        all_trajectories = []
        
        # 1. Generate trajectories
        for query, truth in zip(queries, ground_truths):
            group_trajs = []
            for _ in range(self.config.group_size):
                traj = self.generate_trajectory(query)
                traj.reward = compute_reward(traj, truth)
                group_trajs.append(traj)
            all_trajectories.extend(group_trajs)
            
        # 2. Update policy
        # If we want true gradient accumulation across batches, we would need to 
        # expose optimizer.step() control.
        # However, update_policy internally does optimization steps.
        # To support accumulation, we'd need to modify update_policy to specificy 
        # whether to step or just accumulate.
        # For simplicity in this architecture, we usually assume train_step covers the whole batch.
        # If 'queries' input is a mini-batch, we update immediately.
        
        metrics = self.update_policy(all_trajectories)
        
        # Add basic stats
        tokens = [len(t.token_steps) for t in all_trajectories]
        metrics["avg_tokens"] = np.mean(tokens) if tokens else 0
        metrics["search_trajectories"] = sum(1 for t in all_trajectories if "<search>" in t.generated_text) / len(all_trajectories) if all_trajectories else 0
        
        return metrics
