import json
import pickle
import numpy as np

class MLPClassifier:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.metadata = None
        self.architecture = None
        self.layers = None
        self.weights = []
        self.biases = []
        self.activation_function = None
        self.output_activation_function = None
        self.classes = None
        self.load_model(json_path)
    
    def load_model(self, path: str) -> None:
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
        else:
            with open(path, 'r') as f:
                model_data = json.load(f)
        
        self.metadata = model_data['metadata']
        self.architecture = model_data['architecture']
        self.layers = model_data['layers']
        self.classes = np.array(self.metadata.get('classes', []))
        
        for layer in self.layers:
            self.weights.append(np.array(layer['weights']))
            self.biases.append(np.array(layer['biases']))
        
        self.activation_function = self._get_activation_function(
            self.architecture['activation']
        )
        self.output_activation_function = self._get_activation_function(
            self.architecture['output_activation']
        )
        
        print(f"MLP model loaded from: {path}")
        print(f"Architecture: {self.metadata['n_features']} -> "
              f"{self.architecture['hidden_layer_sizes']} -> "
              f"{self.metadata['n_classes']}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or len(self.weights) == 0:
            raise ValueError("Model is not loaded.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        activation = X
        
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self.activation_function(z)
        
        z_out = np.dot(activation, self.weights[-1]) + self.biases[-1]
        output = self.output_activation_function(z_out)
        
        class_indices = np.argmax(output, axis=1)
        
        if self.classes is not None and len(self.classes) > 0:
            return self.classes[class_indices]
        
        return class_indices
    
    def _get_activation_function(self, activation_name: str):
        activation_functions = {
            'relu': self._relu,
            'tanh': self._tanh,
            'logistic': self._sigmoid,
            'sigmoid': self._sigmoid,
            'identity': self._identity,
            'softmax': self._softmax
        }
        
        return activation_functions[activation_name]
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def _identity(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
