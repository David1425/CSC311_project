import pandas as pd
import numpy as np
from typing import Final
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from helpers.constants import DATA_COLUMNS, LABELS, RAW_DATA_CSV_FP, PP_DATA_CSV_FP
import pickle
import sys
sys.path.append('..')

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
        self.seed = seed if seed is not None else 42
        np.random.seed(self.seed)
        self.bow_matrix = None
        self.vectorizer = None

        self.vocab = {}
        
    def _split_train_val_test(self):
        """
        Split and save all the data into train/val/test
        """
        student_id_col = self.data.columns[DATA_COLUMNS['STUDENT_ID_COLUMN']]
        unique_students = np.unique(self.data[student_id_col])
        
        # SPLIT BY STUDENT IDs
        train_valid_students, test_students = train_test_split(
            unique_students, test_size=0.0325, random_state=self.seed
        )
        
        train_students, val_students = train_test_split(
            train_valid_students, test_size=0.1875, random_state=self.seed
        )
        
        train_mask = np.isin(self.data[student_id_col], train_students)
        val_mask = np.isin(self.data[student_id_col], val_students)
        test_mask = np.isin(self.data[student_id_col], test_students)
        
        train_df_by_student_id = self.data[train_mask]
        train_df_by_student_id.to_csv(RAW_DATA_CSV_FP[0], index=False)
        val_df_by_student_id = self.data[val_mask]
        val_df_by_student_id.to_csv(RAW_DATA_CSV_FP[1], index=False)
        test_df_by_student_id = self.data[test_mask]
        test_df_by_student_id.to_csv(RAW_DATA_CSV_FP[2], index=False)


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
            # Add one-hot encoded columns to self.data, replacing existing selection col.s and empty values with -1.0
            encoded_cols = pd.concat(one_hot_cols, axis=1).reindex(self.data.index, fill_value=0)
            for ec in encoded_cols:
                self.data[ec] = encoded_cols[ec].fillna(-1.0)
    
    
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
        
    def generate_Xt_files(self, fitted_count_vectorizer_path):
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
            with open(fitted_count_vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
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
    
    
    def preprocess(self, filepath):
        """
        Cleans and preprocess a new file
        
        Assumption: Data at the filepath does NOT contain labels
        
        Returns preproccsed features
        """
        selection_cols = self.data.columns[DATA_COLUMNS['SELECTION_COLUMNS']]

        # Read
        self.read_csv(filepath)
        
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
            
        # Return final, clean dataframe
        return self.data