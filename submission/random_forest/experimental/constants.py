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

RF_PATH = 'fitted_bow_rf.pkl'
COUNT_VECTORIZER_PATH = 'count_vectorizer.pkl'
COLUMNS_PATH = 'columns.csv'