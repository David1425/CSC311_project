import numpy as np
import sys
sys.path.append('..')

from constants import PREDICTIONS_TO_LABELS
from random_forest import RandomForestModel
from dataloader import DataLoader

# Setting Config
json_path = 'fitted_bow_rf_cmprsd_std.json'
vocab_path = 'vocab.csv'
columns_path = 'columns.csv'

def predict_all(filename):
    """
    Make predictions for the data in filename. Returns a dataframe with the predictions
    """
    
    # Extract vocab from training cols CSV
    dl = DataLoader()
    dl.extract_vocab_from_columns_csv(columns_path, vocab_path)
    
    # Create new dataloader and load vocab and expected columns 
    dataloader = DataLoader()
    dataloader.load_vocab(vocab_path)
    expected_cols = dataloader.load_column_names(columns_path)
    
    # Process test data with expected columns
    preprocessed_data = dataloader.preprocess(filename, expected_cols)
    
    # Generate model from saved JSON
    rf = RandomForestModel(json_path)
    
    # Make and clean predictions
    base_predictions = rf.predict(preprocessed_data.to_numpy())
    vectorized_map = np.vectorize(PREDICTIONS_TO_LABELS.get)
    return vectorized_map(base_predictions)
    