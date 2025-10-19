import pandas as pd
import numpy as np

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

        self.vocab = {}
    
    def read_csv(self, filepath):
        """
        Read a CSV file and return a pandas DataFrame.
        
        Args:
            filepath: Path to the CSV file.
            
        Returns:
            pandas DataFrame containing the data.
        """

        self.data = pd.read_csv(filepath)
        
    def build_vocab(self, text):
        """
        Build a vocabulary from the given text.

        Args:
            text: Input text to build the vocabulary from.

        Returns:
            A dictionary mapping words to their unique integer IDs.
        """
        