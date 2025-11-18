import pandas as pd
import numpy as np

class BowVectorizer:
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

    def __init__(self, seed=None, truncate_length=4, n_components=50, scale_numeric=True):
        """
        Initialize the vectorizer with an optional rng seed for reproducibility.
        
        Args:
            seed: Random seed. (Optional)
            truncate_length: Maximum length of each word in the text columns.
            n_components: Number of components to keep after SVD compression. (Default: 50)
            scale_numeric: Whether to scale numeric features to match text feature magnitude. (Default: True)
        """
        self.seed = seed
        np.random.seed(self.seed)

        self.truncate_length = truncate_length
        self.n_components = n_components
        self.scale_numeric = scale_numeric
        
        self._word_to_index = {}
        self._svd_components = None  # Will store the SVD transformation matrix
        self._text_mean = None  # Mean for centering before SVD
        self._numeric_scale = 1.0  # Scaling factor for numeric features
    
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

        # Build vocabulary - collect all unique words
        vocab = set()
        
        for idx in self._text_columns:
            col = data.columns[idx]
            for text in data[col]:
                words = text.split()
                for word in words:
                    if word and word.strip():  # Skip empty strings
                        vocab.add(word)
        
        # Build word-to-index mapping for fast lookup
        self._word_to_index = {word: idx for idx, word in enumerate(sorted(vocab))}
    
    def get_bow(self, text, normalize=True):
        """
        Get the Bag of Words vector for a given text.

        Args:
            text: Input text string.
            normalize: Whether to normalize the BoW vector.
        Returns:
            Numpy array representing the BoW vector.
        """
        words = text.split()
        bow_vector = np.zeros(len(self._word_to_index))
        
        if not words:
            return bow_vector
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        for word, count in word_freq.items():
            if word in self._word_to_index:
                index = self._word_to_index[word]
                if normalize:
                    bow_vector[index] = count / len(words)
                else:
                    bow_vector[index] = count
                
        return bow_vector

    def generate_Xt(self, data_path: str, normalize=True, verbose=True):
        """
        Get the text columns from the data.

        Args:
            data_path: Path to the CSV data file.
            normalize: Whether to normalize the BoW vectors.
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
            
            # Get BoW vector and compress with SVD if available
            combined_text = ' '.join([row[data.columns[idx]] for idx in self._text_columns])
            bow_vector = self.get_bow(combined_text, normalize=normalize)
            
            if self._svd_components is not None:
                # Use compressed representation
                compressed_vector = self._compress_bow(bow_vector)
                feature_vector.extend(compressed_vector.tolist())
            else:
                # Use full BoW vector
                feature_vector.extend(bow_vector.tolist())
            
            # Add rating features with optional scaling
            for idx in self._rating_columns:
                col = data.columns[idx]
                if normalize:
                    rating_value = row[col] / 5.0
                else:
                    rating_value = row[col]
                
                # Scale numeric features if enabled
                if self.scale_numeric and self._svd_components is not None:
                    rating_value *= self._numeric_scale
                
                feature_vector.append(rating_value)

            # Add selection features (binary, no scaling needed)
            for idx in self._selection_columns:
                col = data.columns[idx]
                selection_vector = [0] * len(self._selections)
                for selection in row[col]:
                    selection_vector[selection] = 1

                feature_vector.extend(selection_vector)

            X_matrix.append(np.array(feature_vector))

            label_col = data.columns[self._label_column]
            t_vector.append(row[label_col])

        return np.array(X_matrix), np.array(t_vector)
    
    def build_svd(self, data_path: str, normalize=True, verbose=True):
        """
        Build SVD transformation matrix from training data BoW vectors.
        This should be called once on training data before using generate_Xt.

        Args:
            data_path: Path to the CSV data file.
            normalize: Whether to normalize the BoW vectors.
            verbose: Whether to print progress.
        """
        if verbose:
            print(f"Building SVD transformation from {data_path}...")

        data = self.read_csv(data_path, verbose=verbose)

        # Collect all BoW vectors
        bow_matrix = []
        for _, row in data.iterrows():
            combined_text = ' '.join([row[data.columns[idx]] for idx in self._text_columns])
            bow_vector = self.get_bow(combined_text, normalize=normalize)
            bow_matrix.append(bow_vector)
        
        bow_matrix = np.array(bow_matrix)
        
        # Center the data (subtract mean)
        self._text_mean = np.mean(bow_matrix, axis=0)
        centered_matrix = bow_matrix - self._text_mean
        
        # Perform SVD
        if verbose:
            print(f"Performing SVD with n_components={self.n_components}...")
        
        U, S, Vt = np.linalg.svd(centered_matrix, full_matrices=False)
        
        # Keep only top n_components
        n_comp = min(self.n_components, len(S))
        self._svd_components = Vt[:n_comp, :].T  # Shape: (vocab_size, n_components)
        
        if verbose:
            # Calculate variance explained
            total_var = np.sum(S**2)
            explained_var = np.sum(S[:n_comp]**2) / total_var * 100
            print(f"SVD complete. Reduced from {bow_matrix.shape[1]} to {n_comp} dimensions.")
            print(f"Variance explained: {explained_var:.2f}%")
        
        # Calculate scaling factor for numeric features if needed
        if self.scale_numeric:
            # Transform the training data to see typical magnitude
            compressed = np.dot(centered_matrix, self._svd_components)
            text_magnitude = np.mean(np.abs(compressed))
            # Typical rating range is 0-5, we want to scale to similar magnitude as compressed text
            self._numeric_scale = text_magnitude / 0.5  # 0.5 is roughly middle of normalized rating range
            if verbose:
                print(f"Numeric feature scaling factor: {self._numeric_scale:.4f}")
    
    def _compress_bow(self, bow_vector):
        """
        Compress a BoW vector using the learned SVD transformation.

        Args:
            bow_vector: BoW vector to compress.
        Returns:
            Compressed vector.
        """
        if self._svd_components is None:
            raise ValueError("SVD not built. Call build_svd() first on training data.")
        
        # Center and project onto SVD components
        centered = bow_vector - self._text_mean
        compressed = np.dot(centered, self._svd_components)
        return compressed
    
    def get_vocab_size(self):
        """
        Get the size of the vocabulary.

        Returns:
            Integer representing the size of the vocabulary.
        """
        return len(self._word_to_index)