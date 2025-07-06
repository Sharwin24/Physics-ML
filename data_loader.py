import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

MCQ_DATA_PATH = "data/PhysUnivBench_en_MCQ.json"
OE_DATA_PATH = "data/PhysUnivBench_en_OE.json"
UNIFIED_DATA_PATH = "data/PhysUnivBench_en_unified.json"

# MCQ JSON Structure
# {
#     "id":
#     "image":
#     "question":
#     "subtopic":
#     "language":
#     "difficulty":
#     "options":
#     "answer":
#     "parsing":
# }
# OE JSON Structure
# {
#     "id":
#     "image":
#     "question":
#     "subtopic":
#     "language":
#     "difficulty":
#     "answer":
# }
# Unified JSON Structure
# {
#     "id": "",
#     "type": "mcq" | "oe",
#     "image": "<path>",
#     "question": "",
#     "subtopic": "",
#     "difficulty": 0-1, // normalized
#     "options": "", // only for MCQ
#     "answer": "",
#     "parsing": "" // optional metadata
# }


def get_option_in_english(option: str, options: str) -> str:
    # Option is either ["A", "B", "C", "D"]
    # options is a string of form:
    # "A. <text> \n\nB. <text> \n\nC. \n\nD. <text>",
    # This function should return <text> corresponding to the option
    try:
        answer_dict = {}
        options_list = options.split('\n\n')
        for opt in options_list:
            if opt.strip():
                key, value = opt.split('. ', 1)
                answer_dict[key.strip()] = value.strip()
        return answer_dict.get(option.strip(), "")
    except Exception:
        return ""


def create_unified_json(MCQ_DATA_PATH: str, OE_DATA_PATH: str, output_path: str):
    """
    Given 2 JSON files for MCQ and OE datasets, create a single JSON file
    with both datasets where each entry contains a 'type' field as well
    as the original fields.

    Args:
        MCQ_DATA_PATH (str): _description_
        OE_DATA_PATH (str): _description_
        output_path (str): _description_
    """
    mcq_data = pd.read_json(MCQ_DATA_PATH)
    oe_data = pd.read_json(OE_DATA_PATH)

    # Add 'type' column
    mcq_data['type'] = 'mcq'
    oe_data['type'] = 'oe'

    # Normalize difficulty to [0, 1]
    def normalize_difficulty(df):
        if df['difficulty'].max() > 1:
            df['difficulty'] = (df['difficulty'] - df['difficulty'].min()) / \
                               (df['difficulty'].max() - df['difficulty'].min())
        return df

    mcq_data = normalize_difficulty(mcq_data)
    oe_data = normalize_difficulty(oe_data)

    # Force required columns to exist
    required_columns = ['id', 'type', 'image', 'question', 'subtopic',
                        'language', 'difficulty', 'options', 'answer', 'parsing']

    for df in [mcq_data, oe_data]:
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

    # Fill MCQ-specific and OE-specific missing values as empty strings
    mcq_data['parsing'] = mcq_data['parsing'].apply(
        lambda x: x if isinstance(x, str) else "")
    mcq_data['options'] = mcq_data['options'].apply(
        lambda x: x if isinstance(x, str) else "")

    oe_data['options'] = ""  # not used
    oe_data['parsing'] = ""  # optional

    # Select columns in unified order
    mcq_data = mcq_data[required_columns]
    oe_data = oe_data[required_columns]

    # Concatenate
    unified_data = pd.concat([mcq_data, oe_data], ignore_index=True)

    # Ensure difficulty is a float between 0 and 1, and other fields are strings
    unified_data['difficulty'] = unified_data['difficulty'].astype(float)
    for col in required_columns:
        if col != 'difficulty':
            unified_data[col] = unified_data[col].astype(str)

    # Save as a standard JSON list (pretty-printed)
    unified_data.to_json(output_path, orient='records', indent=2)


def create_fine_tuning_json(MCQ_DATA_PATH: str, OE_DATA_PATH: str,
                            mcq_out_path="data/llava_finetune_mcq.json",
                            oe_out_path="data/llava_finetune_oe.json"):
    """
    Converts MCQ and OE datasets into LLaVA fine-tuning format.
    Generates two JSON files with image-text conversation pairs.

    Args:
        MCQ_DATA_PATH: Path to MCQ JSON
        OE_DATA_PATH: Path to OE JSON
        mcq_out_path: Output path for MCQ fine-tune JSON
        oe_out_path: Output path for OE fine-tune JSON
    """
    mcq_data = pd.read_json(MCQ_DATA_PATH)
    oe_data = pd.read_json(OE_DATA_PATH)

    fine_tune_mcq = []
    fine_tune_oe = []

    for _, row in mcq_data.iterrows():
        image_path = row.get("image", "").strip()
        question = row.get("question", "").strip()
        options = row.get("options", "").strip()
        answer_key = row.get("answer", "").strip()
        explanation = get_option_in_english(answer_key, options)

        prompt = f"Which of the following options is correct?\n{options}"
        answer = f"The correct answer is {answer_key} because {explanation}."

        fine_tune_mcq.append({
            "image": image_path,
            "conversations": [
                {"from": "human", "value": question + "\n" + prompt},
                {"from": "gpt", "value": answer}
            ]
        })

    for _, row in oe_data.iterrows():
        image_path = row.get("image", "").strip()
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()

        fine_tune_oe.append({
            "image": image_path,
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer}
            ]
        })

    # Save to disk
    os.makedirs(os.path.dirname(mcq_out_path), exist_ok=True)
    with open(mcq_out_path, "w") as f:
        json.dump(fine_tune_mcq, f, indent=2)

    with open(oe_out_path, "w") as f:
        json.dump(fine_tune_oe, f, indent=2)

    print(
        f"Saved fine-tuning data:\n  MCQ: {mcq_out_path}\n  OE: {oe_out_path}"
    )


def load_data(file_path, test_size=0.2, random_state=42):
    """
    Load data from a unified JSON file and split it into training and testing sets.

    Parameters:
    - file_path: str, path to the JSON file.
    - test_size: float, proportion of the dataset to include in the test split.
    - random_state: int, controls the shuffling applied to the data before applying the split.

    Returns:
    - X_train: DataFrame, training features.
    - X_test: DataFrame, testing features.
    - y_train: Series, training labels.
    - y_test: Series, testing labels.
    """
    # Load the dataset
    data = pd.read_json(file_path)

    # Keep 'type' column for later processing
    X = data[['id', 'type', 'image', 'question', 'subtopic',
              'language', 'difficulty', 'options', 'parsing']].copy()

    # Create labels
    y = []
    for _, row in data.iterrows():
        if row['type'] == 'mcq':
            extracted = get_option_in_english(row['answer'], row['options'])
            y.append(extracted if pd.notnull(extracted) else "")
        elif row['type'] == 'oe':
            y.append(row['answer'])
        else:
            raise ValueError(f"Unknown problem type: {row['type']}")

    y = pd.Series(y, index=data.index, dtype=object)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# Take in MCQ and OE JSON
# Create a Train-Test split
# Convert to a format suitable for LLaVA fine-tuning

if __name__ == "__main__":
    # Create the unified JSON file
    create_unified_json(MCQ_DATA_PATH, OE_DATA_PATH, UNIFIED_DATA_PATH)
    # Create the training and testing sets from the unified JSON file
    X_train, X_test, y_train, y_test = load_data(UNIFIED_DATA_PATH)
    print("Training and testing sets created successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Create fine-tuning JSON files
    create_fine_tuning_json(MCQ_DATA_PATH, OE_DATA_PATH,
                            mcq_out_path="data/llava_finetune_mcq.json",
                            oe_out_path="data/llava_finetune_oe.json")
    print("Fine-tuning JSON files created successfully.")
