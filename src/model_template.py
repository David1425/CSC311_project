from helpers.learning_rate_schedules import *
import numpy as np

# NOTE: If there are parameters that are not used, keep them in the function signature for consistency.

class ModelName:
    """
    A template class for machine learning models.
    """

    # TODO: Put any needed helper functions here
    # e.g., debugging utilities, model-specific methods, forward pass, back propagation etc.

    def __init__(self, ): # TODO: add hyperparameters
        """
        Initialize the model with hyperparameters.
        
        Args:
            **hyperparameters: Arbitrary keyword arguments for model hyperparameters.
                For example:
                - num_hidden_layers (int): Number of hidden layers
                - hidden_size (int): Number of hidden units in each layer
                - regularization (float): Regularization strength
                - activation (str): Activation function type
        """

        # TODO: Perform initialization here
        raise NotImplementedError("__init__ method must be implemented")
        
    def train(self, train_X, train_t, learning_rate, batch_size, n_epochs):
        """
        Train the model on the provided training data.
        
        Args:
            train_X (np.array): Training data of shape (N, num_features)
                where N is the number of data points
            train_t (np.array): Training targets of shape (N, num_classes)
            learning_rate (float or callable): Learning rate as a schedule function
                that takes current epoch and total epochs as input
            batch_size (int): Number of samples per gradient update
            n_epochs (int): Number of training epochs
            
        Returns:
            dict: Training history containing metrics like loss per epoch
        """

        # TODO: Implement training logic here (Can use external libraries if needed)
        raise NotImplementedError("train method must be implemented")
        
    def tune_hyperparameters(self, hyperparameters, train_X, train_t, val_X, val_t):
        """
        Tune model hyperparameters using grid search or similar approach.
        
        Args:
            hyperparameters (dict): Dictionary mapping hyperparameter names (str)
                to lists of values to try.
                Example: {'learning_rate': [constant_lr(0.01), exponential_decay(0.03, 0.95)], 'batch_size': [32, 64]}
            train_X (np.array): Training data of shape (N, num_features)
            train_t (np.array): Training targets of shape (N, num_classes)
            val_X (np.array): Validation data for hyperparameter selection
            val_t (np.array): Validation targets
            
        Returns:
            dict: Best hyperparameters found during tuning
        """

        # TODO: Implement hyperparameter tuning logic here (Can use external libraries if needed)
        raise NotImplementedError("tune_hyperparameters method must be implemented")
        
    def predict(self, X):
        """
        Make predictions on given input data.
        
        Args:
            X (np.array): Input data of shape (N, num_features)
            
        Returns:
            np.array: Predictions of shape (N, num_classes)
        """

        # TODO: Implement prediction logic here
        raise NotImplementedError("predict method must be implemented")
    
    def save_model(self, file_path):
        """
        Save the model parameters to a file.
        
        Args:
            file_path (str): Path to the file where model parameters will be saved.
        """

        # TODO: Implement model saving logic here
        raise NotImplementedError("save_model method must be implemented")
    
    def load_model(self, file_path):
        """
        Load model parameters from a file.
        
        Args:
            file_path (str): Path to the file from which model parameters will be loaded.
        """

        # TODO: Implement model loading logic here
        raise NotImplementedError("load_model method must be implemented")