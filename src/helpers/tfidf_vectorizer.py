import pandas as pd
import numpy as np

class TfidfVectorizer:
    _id_coloumn = 0
    _text_columns = [1, 6, 9]
    _selection_columns = [3, 5]
    _rating_columns = [2, 4, 7, 8]
    _label_column = 10
    _selections = {
        'math computations': 0,
        'writing or debugging code': 1,
        'data processing or analysis': 2,
        'explaining complex concepts simply': 3,
        'converting content between formats': 4,
        'writing or editing essays/reports': 5,
        'drafting professional text': 6,
        'brainstorming or generating creative ideas': 7
    }
    _labels = {
        'ChatGPT': 0,
        'Claude': 1,
        'Gemini': 2
    }

    def __init__(self, seed=None, truncate_length=4):
        """
        Initialize the vectorizer with an optional rng seed for reproducibility.
        
        Args:
            seed: Random seed. (Optional)
            truncate_length: Maximum length of each word in the text columns.
        """
        self.seed = seed
        np.random.seed(self.seed)

        self.truncate_length = truncate_length
        
        # Instance variables instead of class variables
        self._document_count = 0
        self._document_freq = {}
        self._word_to_index = {}  # Cache for fast word lookup
    
    def read_csv(self, filepath, verbose=True):
        """
        Read a CSV file and return a pandas DataFrame.
        
        Args:
            filepath: Path to the CSV file.
            
        Returns:
            pandas DataFrame containing the data.
        """

        if verbose:
            print(f"Reading data from {filepath}...")
            
        self.data = pd.read_csv(filepath)

        if verbose:
            print("Cleaning data...")

        def clean_and_truncate_text(text):
            text = str(text).lower()
            text = text.replace('#name?', '')
            words = []
            for word in text.split():
                cleaned_word = ''.join(c for c in word if c.isalpha())
                if len(cleaned_word) >= 2:
                    truncated_word = cleaned_word[:self.truncate_length]
                    words.append(truncated_word)
            
            return ' '.join(words)

        for idx in self._text_columns:
            col = self.data.columns[idx]
            self.data[col] = self.data[col].apply(clean_and_truncate_text)
        
        for idx in self._rating_columns:
            col = self.data.columns[idx]
            self.data[col] = (
                self.data[col]
                .astype(str)
                .str.extract(r'^\s*(\d+)')
                .fillna(-1)
                .astype(int)
            )
        
        def selection_map(selections):
            if not isinstance(selections, str):
                return []

            selections = selections.lower()
            return [val for key, val in self._selections.items() if key in selections]

        for idx in self._selection_columns:
            col = self.data.columns[idx]
            self.data[col] = (
                self.data[col]
                .astype(str)
                .apply(selection_map)
            )

        col = self.data.columns[self._label_column]
        self.data[col] = (
            self.data[col]
            .astype(str)
            .map(self._labels)
        )

        return self.data

    def build_vocab(self, data_path: str, verbose=True):
        """
        Build vocabulary from the text columns in the data.

        Args:
            data_path: Path to the CSV data file.
        """

        if verbose:
            print(f"Building vocabulary from {data_path}...")

        data = self.read_csv(data_path, verbose=verbose)

        # Count documents correctly - once per row, not per text column
        self._document_count = len(data)
        
        for idx in self._text_columns:
            col = data.columns[idx]
            for text in data[col]:
                words = text.split()
                # Use set to count each word once per document (document frequency)
                unique_words = set(words)
                for word in unique_words:
                    if word and word.strip():  # Skip empty strings
                        if word not in self._document_freq:
                            self._document_freq[word] = 0
                        self._document_freq[word] += 1
        
        # Build word-to-index mapping for fast lookup
        self._word_to_index = {word: idx for idx, word in enumerate(self._document_freq.keys())}
    
    def get_tfidf(self, text, normalize=True):
        """
        Get the TF-IDF vector for a given text.

        Args:
            text: Input text string.
            normalize: Whether to normalize the TF-IDF vector.
        Returns:
            Numpy array representing the TF-IDF vector.
        """
        words = text.split()
        tfidf_vector = np.zeros(len(self._document_freq))
        
        # Handle empty text
        if not words:
            return tfidf_vector
        
        # Build term frequency dictionary (more efficient than repeated count())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate TF-IDF for each unique word
        num_words = len(words)
        for word, count in word_freq.items():
            if word in self._word_to_index:
                tf = count / num_words
                idf = np.log((self._document_count + 1) / (self._document_freq[word] + 1)) + 1
                index = self._word_to_index[word]
                tfidf_vector[index] = tf * idf

        if normalize:
            norm = np.linalg.norm(tfidf_vector)
            if norm > 0:
                tfidf_vector = tfidf_vector / norm
                
        return tfidf_vector

    def generate_Xt(self, data_path: str, normalize=True, verbose=True):
        """
        Get the text columns from the data.

        Args:
            data_path: Path to the CSV data file.
            normalize: Whether to normalize the TF-IDF vectors.
        Returns:
            Numpy array of features and target.
        """

        if verbose:
            print(f"Generating features and targets from {data_path}...")

        data = self.read_csv(data_path, verbose=verbose)

        X_matrix = []
        t_vector = []

        for _, row in data.iterrows():
            feature_vector = []
            
            # for idx in self._text_columns:
            #     col = data.columns[idx]
            #     tfidf_vector = self.get_tfidf(row[col], normalize=normalize)
            #     feature_vector.extend(tfidf_vector.tolist())
            
            combined_text = ' '.join([row[data.columns[idx]] for idx in self._text_columns])
            tfidf_vector = self.get_tfidf(combined_text, normalize=normalize)
            feature_vector.extend(tfidf_vector.tolist())
            
            for idx in self._rating_columns:
                col = data.columns[idx]

                if normalize:
                    feature_vector.extend([row[col]/5.0])
                else:
                    feature_vector.extend([row[col]])

            for idx in self._selection_columns:
                col = data.columns[idx]
                selection_vector = [0]*len(self._selections)
                for selection in row[col]:
                    selection_vector[selection] = 1

                feature_vector.extend(selection_vector)

            X_matrix.append(np.array(feature_vector))

            label_col = data.columns[self._label_column]
            t_vector.append(row[label_col])

        return np.array(X_matrix), np.array(t_vector)
    
    def get_vocab_size(self):
        """
        Get the size of the vocabulary.

        Returns:
            Integer representing the size of the vocabulary.
        """
        return len(self._document_freq)