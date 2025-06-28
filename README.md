# Physics-ML
ML Project from a dataset of physics problems and answers.

# Setting up the virtual env
Run the following to create the virtual environment

```bash
python3 -m venv .venv # Create virtual environment
source .venv/bin/activate # Activate the virtual environment
pip install -r requirements.txt # Install the venv's dependencies
```

# [PhysUniBench](https://huggingface.co/datasets/PrismaX/PhysUniBench) Dataset

For our purposes we have removed the chinese portion of the dataset and will be using the english questions. There are 392 MCQ questions and 628 open-ended questions for a total of **1021 questions**.

**An Undergraduate-Level Physics Reasoning Benchmark for Multimodal Models**

PhysUniBench is the first large-scale multimodal physics benchmark specifically designed for **undergraduate-level understanding, reasoning, and problem-solving**. It provides a valuable testbed for advancing multimodal large language models (MLLMs) with stronger physics reasoning capabilities.

## Key Features

- 📚 **3,304 human-verified multimodal physics problems**, sourced from authentic undergraduate curricula
- 🖼️ Each problem is accompanied by an **image diagram**, enabling joint visual and textual reasoning
- 🌍 **Bilingual coverage**: English and Chinese versions included
- ✏️ **Two question types**: Multiple-Choice Questions (MCQ) and Open-Ended (OE)
- 🧭 Covers **eight core sub-disciplines** of undergraduate physics, with fine-grained difficulty levels

## Dataset Structure

The dataset contains four JSON files and an images folder:
```bash
PhysUniBench/
├── PhysUniBench_en_MCQ.json     # English - Multiple Choice Questions
├── PhysUniBench_en_OE.json      # English - Open-Ended Questions
├── PhysUniBench_zh_MCQ.json     # Chinese - Multiple Choice Questions
├── PhysUniBench_zh_OE.json      # Chinese - Open-Ended Questions
├── images/                      # Folder of all question diagrams
│   ├── 1.png
│   ├── 2.png
│   ├── ...
└── README.md
```

# Pipeline Overview

There are 2 main proposed goals:

1. Physics Problem Solver with a Multimodal interface, using text, image, or both as an input
2. Generate new, diverse, and unique physics problems with correct answers and an accurate difficulty scale

## Data Processing
Unify the data format into a singular JSON observable that can be ingested by a tokenizer compatible with our model. Encode categorical fields (`subtopic`). Resize and normalize images based on model interface (224x224 for ViT or CLIP)

```json
{
  "id": "",
  "type": "mcq" | "oe",
  "image": "<path>",
  "question": "",
  "subtopic": "",
  "difficulty": 0-1, // normalized
  "options": [ ... ], // only for MCQ
  "answer": "",
  "parsing": {} // optional metadata
}
```

*The job of the data_loader is to build a unified dataset using the 2 sub-datasets (MCQ and OE).*

## Multimodal Model Architecture

Ensure our model can accept the format we've defined above (JSON with the unified entries). We can use a Vision-Encoder Language-Decoder (V-Encoder + L-Decoder)

- Image (Vision Encoder), CLIP-ViT, ResNet, etc.
- Text -> Embedding (LLM Encoding), T5, LLaMA, etc.

1. Model Selection
- MiniGPT-4: LLaVA-style, uses ViT-G + Vicuna/LLaMA
- LLaVA: CLIP ViT + LLaMA for VQA and instruction following
- [Fuyu, mPLUG-Owl, or BLIP-2]: Other open multimodal transformers with text-generation

2. Image Encoder
To extract visual features and create visual embeddings (compatible with our LLM)
  - `CLIP-ViT-L/14`
  - `CLIP-ViT-B/32`

3. Text Tokenizer 
To create textual tokens (compatible with our LLM) from input text (questions or prompt)
  - `sentencepiece` (LLaMA or Vicuna)
  - `llama-tokenizer` or HuggingFace's `transformers.AutoTokenizer` for LLaMA

4. Multimodal Fusion Layer
Combines image and text modalities before decoding. 
- Projection Layer: maps vision encoder outputs to LLM embedding dimension
- Cross-attention (image tokens attend to text tokens)
- Concatenation with modality tokens (like `<image>` , `<question>`).

5. Input Formatter
A preprocessor that builds model input from JSON
```python
def format_mcq_prompt(entry):
    q = entry["question"]
    options = entry["options"]
    return f"Question: {q}\nOptions:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
```

6. Training Components
Loss functions could be different based on question type the model is working on:
- MCQ: Cross-entropy over options (classification)
- OE: Seq2Seq language modeling loss

Another training strategy to consider is curriculum learning where the model works on problems sorted from easy to hard since it has access to the difficulty field

Output of the model needs to be one of based on the prompt or the classification of the question:
- MCQ: Classification over the options
- OE: Generative decoder

# References and Resources
- [Inside Multimodal LLaMA 3.2: Understanding Meta’s Vision-Language Model Architecture](https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)