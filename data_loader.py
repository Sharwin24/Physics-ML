import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

MCQ_DATA_PATH = "data/PhysUnivBench_en_MCQ.json"
OE_DATA_PATH = "data/PhysUnivBench_en_OE.json"

# MCQ JSON Structure
# {
#     "id":
#     "image":
#     "question":
#     "subtopic":
#     "language":
#     "difficulty"
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
#     "difficulty"
#     "answer":
# }


def get_option_in_english(option: str, options: str) -> str:
    # Option is either ["A", "B", "C", "D"]
    # options is a string of form:
    # "A. <text> \n\nB. <text> \n\nC. \n\nD. <text>",
    # This function should return <text> corresponding to the option
    answer_dict = {}
    options_list = options.split('\n\n')
    for option in options_list:
        if option.strip():
            key, value = option.split('. ', 1)
            answer_dict[key.strip()] = value.strip()
    return answer_dict.get(option.strip(), np.nan)


def load_data(file_path, test_size=0.2, random_state=42):
    """
    Load data from a JSON file and split it into training and testing sets.

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

    # Check if the dataset is MCQ or OE based on the presence of 'options'
    if 'options' in data.columns:
        # MCQ dataset
        X = data[['id', 'image', 'question', 'subtopic',
                  'language', 'difficulty', 'options']]
        y = get_option_in_english(data['answer'])
    else:
        # OE dataset
        X = data[['id', 'image', 'question',
                  'subtopic', 'language', 'difficulty']]
        y = data['answer']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    for path in [MCQ_DATA_PATH, OE_DATA_PATH]:
        print(f"Loading data from {path}...")
        X_train, X_test, y_train, y_test = load_data(path)

        print("\tTraining features shape:", X_train.shape)
        print("\tTesting features shape:", X_test.shape)
        print("\tTraining labels shape:", y_train.shape)
        print("\tTesting labels shape:", y_test.shape)
