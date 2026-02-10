# Search-R1: Agentic RL Implementation

This project implements Search-R1, a Reinforcement Learning framework based on GRPO (Generalized Reward Policy Optimization) that enables large language models to autonomously initiate searches during step-by-step reasoning.

## Overview

Search-R1 introduces a novel approach where LLMs learn when and how to search for information during reasoning, solving the problem of passive search execution in traditional RAG and tool-calling methods.

## Key Features

- **Autonomous Search**: Models learn to decide when to search during reasoning
- **GRPO Algorithm**: Implements Group-based Policy Optimization for stable training
- **Result-Oriented Rewards**: Simple reward design that only checks if the answer is correct
- **Format Validation**: Ensures proper use of `<think>`, `<search>`, `<information>`, and `<answer>` tags

## Project Structure

```
.
├── src/                  # Source code
│   ├── agent/            # Agent components (model, search engine)
│   ├── core/             # Core logic (GRPO, trajectory)
│   ├── data/             # Data loading
│   ├── rewards/          # Reward functions
│   ├── utils/            # Utilities (config)
│   └── run.py            # Main entry point
├── scripts/              # Helper scripts and OpenAI evaluation
│   ├── run_openai.py
│   └── trainer_openai.py
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── ...
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Model

You have two options:

#### Option A: Use OpenAI API (Recommended for evaluation)

**IMPORTANT**: For security, always use environment variables for API keys:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

The configuration in `searchr1_config.py` will automatically use the environment variable:
```python
use_openai: bool = True
openai_api_key: str = os.getenv("OPENAI_API_KEY", "")  # Loads from environment
openai_model: str = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo", etc.
```

**Note**: OpenAI models cannot be updated, so this is for evaluation only. Use `run_openai.py` for evaluation.

#### Option B: Use Local Model (Required for full RL training)

#### Option B: Use Local Model (Required for full RL training)

Modify `src/utils/config.py` or pass arguments (implementation pending):
```python
@dataclass
class SearchR1Config:
    model_name_or_path: str = "your-model-path-here"  # Modify in src/utils/config.py
```

**VRAM Requirements:**
- `qwen3-0.6b`: At least 20GB VRAM (performance will be poor)
- `qwen3-4b` or higher: At least 40GB VRAM (recommended)

### 3. Run Training/Evaluation

**For OpenAI API (evaluation only):**
```bash
python run_openai.py
```

**For Local Model (full RL training):**
```bash
python -m src.run
```

## Important Note: OpenAI vs Local Models

**OpenAI API Models:**
- ✅ Can generate trajectories with log probabilities
- ✅ Can compute rewards and advantages
- ❌ **Cannot be updated** (models are not fine-tunable via API)
- Use `trainer_openai.py` and `run_openai.py` for evaluation

**Local Models:**
- ✅ Full GRPO training with parameter updates
- ✅ Can learn from rewards and improve over time
- ❌ Requires significant VRAM
- Use `trainer.py` and `run.py` for training

For full RL training, you need a local fine-tunable model. OpenAI API is useful for:
- Evaluating the reward function
- Testing the search mechanism
- Generating trajectories for analysis
- Computing advantages for analysis

## Training Process

The training follows these steps:

1. **Generate Answers and Save Trajectories**: For each query, generate `group_size=2` answers and record the generation trajectory
2. **Calculate Rewards**: Compute format rewards and answer correctness rewards (result-oriented)
3. **Calculate Intra-group Advantage**: Normalize rewards within each group: `A_i = (r_i - μ) / σ`
4. **Update Model**: Use GRPO algorithm to update the policy based on advantages
5. **Statistics**: Track metrics like average tokens, search trajectory ratio, etc.

## Reward Design

The reward function is result-oriented and simple:
- **Format Reward**: +0.5 if format is correct, -1.0 if incorrect
- **Answer Reward**: 
  - +2.0 for exact match
  - +1.0 for similarity > 0.5
  - +0.5 for "no content found" response
  - 0.0 otherwise

## GRPO Algorithm

The implementation uses GRPO (Generalized Reward Policy Optimization) with:
- Group-internal advantage calculation
- PPO-style clipping for stable updates
- KL divergence regularization
- Importance sampling for off-policy correction

## Dataset Format

The dataset uses a simple format:
- `queries`: List of training questions
- `ground_truths`: List of corresponding correct answers

Example from school regulations:
```python
queries = [
    "学校对入学新生的复查应在多长时间内完成?",
    "留校察看处分的期限一般是多久?",
]
ground_truths = [
    "学生入学后,学校应当在3个月内按国家招生规定复查...",
    "除开除学籍外,警告、严重警告、记过、留校察看处分一般设置6-12个月期限...",
]
```

## Search Engine

The default implementation uses keyword-based search. In practical scenarios, you can:
- Replace with web search (e.g., using search APIs)
- Replace with RAG search (vector database retrieval)
- Customize the `SearchEngine` class in `search_engine.py`

## Output Format

The model is trained to generate responses with the following structure:

```
<think>思考过程</think>
<search>搜索查询</search>
<information>检索到的信息</information>
<answer>最终答案</answer>
```

## Practical Tips

1. **SFT First**: If direct RL doesn't work well (e.g., format issues), first use Supervised Fine-Tuning (SFT) for instruction following, then proceed with RL
2. **Customize Dataset**: Adjust dataset size and format according to your actual data
3. **Customize Rewards**: The reward function can be completely customized
4. **Use Larger Models**: If possible, use larger and newer models for better performance
5. **API Access**: For production use, consider calling API-based large models for stronger agent capabilities

## Notes

- This project focuses on reproducing the Search-R1 process and implementing GRPO from scratch
- The implementation follows the core training steps; specific details can be debugged and learned independently
- For research purposes, consider extending the implementation with additional features

## References

- Paper: [Search-R1](https://arxiv.org/pdf/2503.09516)
- Project: [Search-R1 GitHub](https://github.com/PeterGriffinJin/Search-R1)
