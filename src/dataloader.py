import pandas as pd
import numpy as np
from typing import Final
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

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

class DataLoader:
    """
    DataLoader class for reading and preprocessing data.
    """

    def __init__(self, seed=None):
        """
        Initialize the DataLoader with a seed for reproducibility.
        
        Args:
            seed: Random seed. If None, a random seed will be generated.
        """
        self.seed = seed
        np.random.seed(self.seed)
        self.bow_matrix = None
        self.vectorizer = None

        self.vocab = {}
        
    def _split_train_val_test(self):
        """
        Split and save all the data into train/val/test
        """
        label_col = self.data.columns[DATA_COLUMNS['LABEL_COLUMN']]
        train_and_valid_df, test_df = train_test_split(
            self.data,
            test_size=0.0325,
            stratify=self.data[label_col],
            random_state=self.seed
        )
        
        train_df, valid_df = train_test_split(
            train_and_valid_df,
            test_size=0.1875,
            stratify=train_and_valid_df[label_col],
            random_state=self.seed
        )
        
        train_df.to_csv(RAW_DATA_CSV_FP[0], index=False)
        valid_df.to_csv(RAW_DATA_CSV_FP[1], index=False)
        test_df.to_csv(RAW_DATA_CSV_FP[2], index=False)


    def _generate_mcq_col_feats(self):
        """
        Generate features from mcq column indices, mutating self.data
        """
        mcq_cols = self.data.columns[DATA_COLUMNS['MCQ_COLUMNS']]
        
        for col in mcq_cols:
            self.data[col] = self.data[col].str[0]
            # Set all col values to be floats, filling missing values with -1.0 as it is not in the possible range
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce', downcast='float').fillna(-1.0)
            
    
    def _safe_split(self, text):
        if pd.isna(text):
            return []
        parts = re.split(r',\s*(?![^()]*\))', str(text))
        return [p.strip() for p in parts if p.strip()]
    
        
    def _generate_selection_col_feats(self):
        """
        Generates features from selection column indices, mutating self.data
        """
        selection_cols = self.data.columns[DATA_COLUMNS['SELECTION_COLUMNS']]
        one_hot_cols = []
                
        for col in selection_cols:
            # Generate temp values using all selected values in row
            temp = (
                self.data[[col]]
                .assign(value=self.data[col].apply(self._safe_split))
                .explode('value')
            )
            
            # Omit all missing values
            temp = temp[temp['value'] != '']
            
            # Early return if no values
            if temp.empty:
                continue
            
            # Generate one-hot encoded values
            one_hot = (
                pd.crosstab(temp.index, temp['value'])
                .astype(int)
                .add_prefix(f"{col}: ")
            )
                        
            one_hot_cols.append(one_hot)
            
        if one_hot_cols:
            # Add one-hot encoded columns to self.data, replacing existing selection col.s
            encoded_cols = pd.concat(one_hot_cols, axis=1).reindex(self.data.index, fill_value=0)
            for ec in encoded_cols:
                self.data[ec] = encoded_cols[ec]
    
    
    def read_csv(self, filepath):
        """
        Read a CSV file and return a pandas DataFrame.
        
        Args:
            filepath: Path to the CSV file.
            
        Returns:
            pandas DataFrame containing the data.
        """

        self.data = pd.read_csv(filepath)
        return self.data
    
        
    def build_vocab(self, text):
        """
        Build a vocabulary from the given text.

        Args:
            text: Input text to build the vocabulary from.

        Returns:
            A dictionary mapping words to their unique integer IDs.
        """
        if self.vectorizer:
                return self.vectorizer.vocabulary_
            
        self.vectorizer = CountVectorizer(stop_words='english')
        self.bow_matrix = self.vectorizer.fit_transform(text)
        self.vocab = self.vectorizer.vocabulary_
        return self.vocab
        
    def generate_Xt_files(self):
        """
        Creates a new .csv file containing the cleaned, pre-processed data
        """
        label_col = self.data.columns[DATA_COLUMNS['LABEL_COLUMN']]
        selection_cols = self.data.columns[DATA_COLUMNS['SELECTION_COLUMNS']]
        
        # Split data first into train/valid/test and save into separate CSVs to prevent data leakage  
        self._split_train_val_test()
                
        # For each CSV
        for i in range(len(RAW_DATA_CSV_FP)):
            # Read
            self.read_csv(RAW_DATA_CSV_FP[i])
            
            # Clean text data
            text_cols = self.data.columns[DATA_COLUMNS['TEXT_COLUMNS']]
            self.data[text_cols] = self.data[text_cols].fillna('').astype(str).apply(lambda x: x.str.lower())
            self.data['large text'] = self.data[text_cols[0]].str.cat(self.data[text_cols[1:2]], sep=' ')
                
            # Generate features from MCQ columns 
            self._generate_mcq_col_feats()
            
            # Generate features from selection columns
            self._generate_selection_col_feats()
            
            # Build vocab and bow matrix
            self.build_vocab(self.data['large text'])
            bow_df = pd.DataFrame(self.bow_matrix.toarray(), columns=self.vectorizer.get_feature_names_out()) # type: ignore
            
            # Drop old text and selection columns and add vocab features
            self.data = self.data.drop(columns=selection_cols).drop(columns=text_cols).drop(columns=['large text'])
            self.data = pd.concat([self.data, bow_df], axis=1)
            self.data = self.data.dropna(subset=[self.data.columns[DATA_COLUMNS['STUDENT_ID_COLUMN']]])
            
            # Move label to end and encode
            self.data = self.data[[col for col in self.data.columns if col != label_col] + [label_col]]
            self.data[label_col] = self.data[label_col].map(LABELS)
            
            # Save as new
            self.data.to_csv(PP_DATA_CSV_FP[i])
            
            
    def generate_Xt(self, filepath):
        """
        Generates X, t from the given filepath.
        
        Assumption: Data at the filepath is cleaned and pre-processed
        
        Returns X, t
        """
        # Read cleaned, pre-processed data
        self.read_csv(filepath=filepath)
        
        return self.data.iloc[:, :-1], self.data.iloc[:, -1:]