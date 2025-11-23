from typing import Final 

DATA_COLUMNS: Final = {
    'STUDENT_ID_COLUMN': 0,
    'TEXT_COLUMNS': [1, 6, 9],
    'MCQ_COLUMNS':  [2, 4, 7, 8],
    'SELECTION_COLUMNS': [3, 5],
    'LABEL_COLUMN': 10   
}

LABELS: Final = {
    'ChatGPT': 0, 
    'Claude': 1, 
    'Gemini': 2
}

PREDICTIONS_TO_LABELS: Final = {
    0: 'ChatGPT',
    1: 'Claude',
    2: 'Gemini'
}

SELECTIONS: Final = {
    'math computations': 0,
    'writing or debugging code': 1,
    'data processing or analysis': 2,
    'explaining complex concepts simply': 3,
    'converting content between formats': 4,
    'writing or editing essays/reports': 5,
    'drafting professional text': 6,
    'brainstorming or generating creative ideas': 7
}

RAW_DATA_CSV_FP: Final = ["./train_data_raw.csv", "./validation_data_raw.csv", "./test_data_raw.csv"]
PP_DATA_CSV_FP: Final = ["./train_data.csv", "./validation_data.csv", "./test_data.csv"]

DL_STD_RAW_DATA_CSV_FP: Final = ["./train_data_raw_std.csv", "./validation_data_raw_std.csv", "./test_data_raw_std.csv"]
DL_STD_PP_DATA_CSV_FP: Final = ["./train_data_std.csv", "./validation_data_std.csv", "./test_data_std.csv"]