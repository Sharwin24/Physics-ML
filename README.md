# Physics-ML
ML Project from a dataset of physics problems and answers.


# Dataset (PhysUniBench)

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
```
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