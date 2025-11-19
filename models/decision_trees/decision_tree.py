import json
import numpy as np

class DecisionTreeModel:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.tree = None
        self.metadata = None
        self.load_tree(json_path)
    
    def load_tree(self, path: str) -> None:
        with open(path, 'r') as f:
            tree_data = json.load(f)
        
        self.metadata = tree_data['metadata']
        self.tree = tree_data['tree']
        print(f"Tree loaded from: {path}")
    
    def _traverse_tree(self, node: dict, sample: np.ndarray):
        if node['is_leaf']:
            return node['predicted_class']
        
        feature_idx = node['feature']
        feature_value = sample[feature_idx]
        threshold = node['threshold']
        
        if feature_value <= threshold:
            return self._traverse_tree(node['left'], sample)
        else:
            return self._traverse_tree(node['right'], sample)
    
    def predict(self, X: np.ndarray):
        if self.tree is None:
            raise ValueError("Tree is not loaded.")
        
        if X.ndim == 1:
            return np.array([self._traverse_tree(self.tree, X)])
        
        predictions = np.array([self._traverse_tree(self.tree, sample) for sample in X])
        return predictions