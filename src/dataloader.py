import pandas as pd
import numpy as np

class DataLoader:
    """
    DataLoader class for reading and preprocessing data.
    """

    _id_coloumn = 0
    _text_columns = [1, 6, 9]
    _selection_columns = [3, 5]
    _choice_columns = [2, 4, 7, 8]
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

    _document_count = 0
    _document_freq = {}

    def __init__(self, seed=None, truncate_length=4):
        """
        Initialize the DataLoader with a seed for reproducibility.
        
        Args:
            seed: Random seed. If None, a random seed will be generated.
        """
        self.seed = seed
        np.random.seed(self.seed)

        self.truncate_length = truncate_length
    
    def read_csv(self, filepath):
        """
        Read a CSV file and return a pandas DataFrame.
        
        Args:
            filepath: Path to the CSV file.
            
        Returns:
            pandas DataFrame containing the data.
        """

        print(f"Reading data from {filepath}...")
        self.data = pd.read_csv(filepath)

        print("Cleaning data...")

        def truncate_text(text):
            return ' '.join([word[:self.truncate_length] for word in text.split()])

        for idx in self._text_columns:
            col = self.data.columns[idx]
            self.data[col] = (
                self.data[col]
                .astype(str)
                .str.lower()
                .str.replace('#name?', '', regex=False)
                .str.replace(r'[^a-z]', ' ', regex=True)
                .str.strip()
                .apply(truncate_text)
                .str.replace(r'\s+', ' ', regex=True)
            )
        
        for idx in self._choice_columns:
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
    
    def split_data(self, train_size, val_size, data=None):
        """
        Split the data into training, validation, and test sets. Labels are equally distributed.

        Args:
            data: pandas DataFrame containing the data.
            train_size: Amount of data to use for training.
            val_size: Amount of data to use for validation.
            data: pandas DataFrame containing the data. If None, uses self.data.

        Returns:
            Tuple of (train_data, val_data, test_data).
        """

        if data is None:
            data = self.data

        train_list, val_list, test_list = [], [], []

        for label, group in self.data.groupby(self.data.columns[self._label_column]):
            group = group.sample(frac=1, random_state=self.seed).reset_index(drop=True)

            train_list.append(group.iloc[:train_size])
            val_list.append(group.iloc[train_size:train_size + val_size])
            test_list.append(group.iloc[train_size + val_size:])

        self.train_data = pd.concat(train_list).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.val_data = pd.concat(val_list).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.test_data = pd.concat(test_list).sample(frac=1, random_state=self.seed).reset_index(drop=True)

        return self.train_data, self.val_data, self.test_data

    def build_vocab(self, data=None):
        """
        Build vocabulary from the text columns in the data.

        Args:
            data: pandas DataFrame containing the data. If None, uses self.data.
        """

        if data is None:
            data = self.data

        for idx in self._text_columns:
            col = data.columns[idx]
            for text in data[col]:
                words = text.split()
                for word in words:
                    if word not in self._document_freq:
                        self._document_freq[word] = 0
                    self._document_freq[word] += 1
                
                self._document_count += 1
    
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

        for word in words:
            if word in self._document_freq:
                tf = words.count(word) / len(words)
                idf = np.log((self._document_count + 1) / (self._document_freq[word] + 1)) + 1
                index = list(self._document_freq.keys()).index(word)
                tfidf_vector[index] = tf * idf

        if np.linalg.norm(tfidf_vector) > 0 and normalize:
            tfidf_vector = tfidf_vector / np.linalg.norm(tfidf_vector)
        return tfidf_vector

    def get_Xt(self, data=None, normalize=True):
        """
        Get the text columns from the data.

        Args:
            data: pandas DataFrame containing the data. If None, uses self.data.
        Returns:
            Numpy array of features and target.
        """

        if data is None:
            data = self.data

        X_matrix = []
        t_vector = []

        for _, row in data.iterrows():
            feature_vector = []
            for idx in self._text_columns:
                col = data.columns[idx]
                tfidf_vector = self.get_tfidf(row[col], normalize=normalize)
                feature_vector.extend(tfidf_vector.tolist())
            
            for idx in self._choice_columns:
                col = data.columns[idx]
                one_hot = [0]*5
                if row[col] >= 1 and row[col] <= 5:
                    one_hot[row[col]-1] = 1

                feature_vector.extend(one_hot)

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