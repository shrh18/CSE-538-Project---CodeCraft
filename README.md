# CodeCraft: Leveraging LLMs for Automated Code Generation

## Introduction
CodeCraft aims to revolutionize software development by leveraging Large Language Models (LLMs) for automated code generation. This project fine-tunes an LLM on a comprehensive corpus of LeetCode problems and solutions, enabling it to generate high-quality code implementations.

## Motivation
Developers face significant challenges in solving coding problems efficiently. By automating the code generation process through NLP and LLMs, CodeCraft aims to enhance productivity and streamline software development, from prototyping to deployment.

## Dataset
The project uses the `greengerong/leetcode` dataset from Hugging Face, comprising ~2,400 LeetCode problem-solution pairs in Python. The dataset covers various topics (arrays, dynamic programming, etc.) and includes metadata on difficulty levels.

### Dataset Summary:
- **Source:** LeetCode
- **Content:** Problem descriptions, input/output examples, code solutions, metadata
- **Size:** ~2,400 pairs
- **Language:** Python
- **Metadata:** Difficulty levels, topics, hints, tags

## Methodology
### 1. Model and Data Selection
- **Model:** Code Llama 2
- **Data:** Converted to Pandas DataFrame for manipulation, then back to Hugging Face datasets format.

### 2. Tokenization and Model Configuration
- **Tokenizer:** AutoTokenizer from Hugging Face
- **Model:** AutoModelForCausalLM with 4-bit quantization using BitsAndBytesConfig

### 3. Parameter-Efficient Fine-Tuning (PEFT)
- **Library:** peft for Low-Rank Adaptation (LoRA)
- **Configuration:** 
  - `r` Parameter: Rank of low-rank matrix
  - Alpha: Scaling factor
- **Approach:** Freezing original model parameters and training injected matrices

### 4. Training Process
- **Trainer:** SFTTrainer from trl library
- **Hyperparameters:** 
  - Batch Size: 2 per device
  - Gradient Accumulation: 4 steps
  - Optimizer: paged_adamw_32bit
  - Learning Rate: 2e-4
  - Scheduler: cosine
  - FP16: Enabled

### 5. Model Training Execution
- **Sequence Length:** Max 512 tokens
- **Training Epochs:** Single epoch with 100 steps

### 6. Inference and Validation
- **Pipeline:** Text generation using Hugging Face Transformers
- **Parameters:** Temperature and top_p for diversity and sampling

## Results
Performance was evaluated using BLEU and CodeBLEU metrics across different problem difficulties (Easy, Medium, Hard).

### Summary of Results:
| Difficulty | BLEU | CodeBLEU |
|------------|------|----------|
| Easy       | 0.85 | 0.78     |
| Medium     | 0.65 | 0.60     |
| Hard       | 0.45 | 0.40     |

### Insights:
- **Easy Problems:** High similarity to reference solutions.
- **Medium Problems:** Satisfactory performance but lacked optimization.
- **Hard Problems:** Struggled with advanced logic.

## Conclusion
CodeCraft showcases the potential of LLMs in automating code generation, significantly enhancing software development efficiency. While effective for simpler problems, more work is needed for complex tasks. Future improvements include dataset expansion, feedback loop integration, and refined learning strategies.

## References
1. OpenAI, "OpenAI Codex: Powering GitHub Copilot"
2. MDPI Editorial, "Google AlphaCode: Competing with the Coding Elite"
3. ACL Anthology, "CodeT5+: Enhancing Code Generation Models"
4. Papers With Code, "GPT-4 and In-Context Learning for Code Generation"
5. Hugging Face, "Datasets: A Hub of Public Datasets for Machine Learning"
6. LeetCode, "Explore Coding Challenges and Solutions"
7. Hu, E. J. et al., "LoRA: Low-Rank Adaptation of Large Language Models"
8. Touvron, H. et al., "LLaMA: Open and Efficient Foundation Language Models"
9. Rozière, B. et al., "Code Llama: Open Foundation Models for Code"
