import pandas as pd
import numpy as np
from typing import Final
import re
from collections import Counter
from constants import DATA_COLUMNS, STOP_WORDS

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

    def _tokenize(self, text):
        """
        Tokenize text into words, removing punctuation and stop words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words (keep only alphanumeric)
        words = re.findall(r'\b[a-z0-9]+\b', text)
        
        # Remove stop words
        tokens = [word for word in words if word not in STOP_WORDS]
        
        return tokens

    def _generate_mcq_col_feats(self):
        """
        Generate features from mcq column indices, mutating self.data
        """
        mcq_cols = self.data.columns[DATA_COLUMNS['MCQ_COLUMNS']]
        
        for col in mcq_cols:
            # Ensure column is string type before using .str accessor
            self.data[col] = self.data[col].astype(str).str[0]
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
            
            # Generate one-hot encoded values as floats to match sklearn
            one_hot = (
                pd.crosstab(temp.index, temp['value'])
                .astype(float)
                .add_prefix(f"{col}: ")
            )
                        
            one_hot_cols.append(one_hot)
            
        if one_hot_cols:
            # Add one-hot encoded columns to self.data, replacing existing selection col.s and empty values with -1.0
            encoded_cols = pd.concat(one_hot_cols, axis=1).reindex(self.data.index, fill_value=0.0)
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
        Build a vocabulary from the given text and create BoW matrix.

        Args:
            text: Input text (pandas Series or list of strings) to build the vocabulary from.

        Returns:
            A dictionary mapping words to their unique integer IDs.
        """
        if self.vocab:
            # If vocab already exists, just transform the text
            return self.vocab
        
        # Handle pandas Series or list
        if isinstance(text, pd.Series):
            documents = text.tolist()
        else:
            documents = text
        
        # Collect all tokens from all documents
        all_tokens = []
        for doc in documents:
            tokens = self._tokenize(str(doc))
            all_tokens.extend(tokens)
        
        # Create vocabulary: assign unique ID to each unique word (sorted for consistency)
        unique_words = sorted(set(all_tokens))
        self.vocab = {word: idx for idx, word in enumerate(unique_words)}
        
        # Create BoW matrix using the vocabulary
        self.bow_matrix = self._transform_with_vocab(documents)
        
        return self.vocab
    
    def _transform_with_vocab(self, documents):
        """
        Transform documents into BoW matrix using existing vocabulary.
        
        Args:
            documents: List of text documents
            
        Returns:
            BoW matrix as numpy array
        """
        # Create BoW matrix
        n_docs = len(documents)
        n_vocab = len(self.vocab)
        bow_matrix = np.zeros((n_docs, n_vocab), dtype=int)
        
        # Fill the BoW matrix
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(str(doc))
            token_counts = Counter(tokens)
            
            for word, count in token_counts.items():
                if word in self.vocab:
                    word_idx = self.vocab[word]
                    bow_matrix[doc_idx, word_idx] = count
        
        return bow_matrix
    
    def save_vocab(self, filepath):
        """
        Save the vocabulary to a CSV file.
        
        Args:
            filepath: Path to save the vocabulary CSV
        """
        if not self.vocab:
            raise ValueError("No vocabulary to save. Build vocabulary first.")
        
        # Create a DataFrame with vocabulary words
        vocab_df = pd.DataFrame({'word': list(self.vocab.keys())})
        vocab_df.to_csv(filepath, index=False)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath):
        """
        Load vocabulary from a CSV file.
        
        Args:
            filepath: Path to the vocabulary CSV file (one word per row with 'word' header)
            
        Returns:
            A dictionary mapping words to their unique integer IDs.
        """
        # Read the vocabulary file
        vocab_df = pd.read_csv(filepath)
        
        # Words are in the 'word' column
        words = vocab_df['word'].tolist()
        
        # Rebuild the vocabulary dictionary with consistent ordering
        self.vocab = {word: idx for idx, word in enumerate(words)}
        
        print(f"Vocabulary loaded from {filepath} ({len(self.vocab)} words)")
        return self.vocab
    
    def extract_vocab_from_columns_csv(self, columns_filepath, output_vocab_filepath, start_col_index=22):
        """
        Extract vocabulary words from a columns CSV file and save to vocab format.
        
        This is useful when you have a columns.csv with all feature names but need to
        extract just the vocabulary words into a separate vocab.csv file.
        
        Args:
            columns_filepath: Path to the CSV file containing all column names
            output_vocab_filepath: Path to save the extracted vocabulary
            start_col_index: Index where vocabulary columns start (default 20, after non-BoW features)
        """
        # Read the columns file
        columns_df = pd.read_csv(columns_filepath)
        
        # Get all column names
        all_columns = columns_df.columns.tolist()
        
        # Extract vocabulary words (everything after start_col_index, excluding 'label' if present)
        vocab_words = []
        for i, col in enumerate(all_columns):
            if i >= start_col_index and col.lower() != 'label':
                vocab_words.append(col)
        
        # Remove the first empty string if present
        if vocab_words and vocab_words[0] == '':
            vocab_words = vocab_words[1:]
        
        # Save to vocab format (one word per row)
        vocab_df = pd.DataFrame({'word': vocab_words})
        vocab_df.to_csv(output_vocab_filepath, index=False)
        
        print(f"Extracted {len(vocab_words)} vocabulary words")
        print(f"Vocabulary saved to {output_vocab_filepath}")
        
        return vocab_words
    
    def get_feature_names_out(self):
        """
        Get feature names (vocabulary words) in the same order as matrix columns.
        
        Returns:
            List of feature names
        """
        if not self.vocab:
            return []
        
        # Sort by index to get words in column order
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        return [word for word, idx in sorted_vocab]
    
    def preprocess(self, filepath, expected_columns=None):
        """
        Cleans and preprocess a new file
        
        Assumption: Data at the filepath does NOT contain labels
        
        Args:
            filepath: Path to the CSV file
            expected_columns: List of expected column names (for ensuring consistency with training)
        
        Returns preprocessed features
        """
        # Read
        self.read_csv(filepath)
        
        selection_cols = self.data.columns[DATA_COLUMNS['SELECTION_COLUMNS']]
        
        # Clean text data
        text_cols = self.data.columns[DATA_COLUMNS['TEXT_COLUMNS']]
        self.data[text_cols] = self.data[text_cols].fillna('').astype(str).apply(lambda x: x.str.lower())
        self.data['large text'] = self.data[text_cols[0]].str.cat(self.data[text_cols[1:2]], sep=' ')
            
        # Generate features from MCQ columns 
        self._generate_mcq_col_feats()
        
        # Generate features from selection columns
        self._generate_selection_col_feats()
        
        # Build vocab and bow matrix (or use existing vocab)
        if not self.vocab:
            self.build_vocab(self.data['large text'])
        else:
            # Use existing vocabulary to transform
            test_docs = self.data['large text'].tolist()
            self.bow_matrix = self._transform_with_vocab(test_docs)
        
        # Convert bow matrix to array
        bow_array = self.bow_matrix
            
        bow_df = pd.DataFrame(bow_array, columns=self.get_feature_names_out())
        
        # Drop old text and selection columns and add vocab features
        self.data = self.data.drop(columns=selection_cols).drop(columns=text_cols).drop(columns=['large text'])
        self.data = pd.concat([self.data, bow_df], axis=1)
        self.data = self.data.dropna(subset=[self.data.columns[DATA_COLUMNS['STUDENT_ID_COLUMN']]])
        
        # If expected columns are provided, align the dataframe
        if expected_columns is not None:
            # Add missing columns with 0s
            for col in expected_columns:
                if col not in self.data.columns:
                    self.data[col] = 0
            
            # Keep only expected columns in the correct order
            self.data = self.data[expected_columns]
            
        # Return final, clean dataframe
        return self.data
    
    def save_column_names(self, filepath):
        """
        Save the column names (feature names) after preprocessing.
        
        Args:
            filepath: Path to save the column names CSV
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data to extract column names from. Run preprocess() first.")
        
        # Create a DataFrame with column names
        columns_df = pd.DataFrame({'column': self.data.columns.tolist()})
        columns_df.to_csv(filepath, index=False)
        print(f"Column names saved to {filepath}")
    
    def load_column_names(self, filepath):
        """
        Load expected column names from a CSV file.
        
        Args:
            filepath: Path to the column names CSV file (column names are in header row)
            
        Returns:
            List of column names
        """
        # Read the columns file with index_col=0 to match how read_csv works
        columns_df = pd.read_csv(filepath, index_col=0)
        
        # Column names are in the header (after treating first column as index)
        column_names = columns_df.columns.tolist()
        
        # Remove the first empty column if it exists
        if column_names and column_names[0] == '':
            column_names = column_names[1:]
        
        print(f"Column names loaded from {filepath} ({len(column_names)} columns)")
        return column_names
    

    def generate_Xt(self, filepath):
        """
        Generates X, t from the given filepath.
        
        Assumption: Data at the filepath is cleaned and pre-processed
        
        Returns X, t
        """
        # Read cleaned, pre-processed data
        self.read_csv(filepath=filepath)
        
        return self.data.iloc[:, :-1], self.data.iloc[:, -1:]