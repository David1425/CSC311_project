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

# Common English stop words (equivalent to sklearn's 'english' stop words)
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

RAW_DATA_CSV_FP: Final = ["./train_data_raw.csv", "./validation_data_raw.csv", "./test_data_raw.csv"]
PP_DATA_CSV_FP: Final = ["./train_data.csv", "./validation_data.csv", "./test_data.csv"]