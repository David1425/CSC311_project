import pickle
import numpy as np
from collections import Counter

class RandomForestModel:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.trees = None
        self.metadata = None
        self.n_estimators = None
        self.load_forest(json_path)
    
    def load_forest(self, path: str) -> None:
        with open(path, 'rb') as f:
            forest_data = pickle.load(f)
            
        self.metadata = forest_data['metadata']
        self.trees = forest_data['trees']
        self.n_estimators = self.metadata['n_estimators']
    
    def _traverse_tree(self, node: dict, sample: np.ndarray):
        """Traverse a single tree to get prediction with sample weight."""
        if node['is_leaf']:
            # Return both the predicted class probabilities (as weights)
            values = np.array(node['value']).flatten()
            # Return the proportion in each class (this is the weight)
            return values / values.sum()
        
        feature_idx = node['feature_index']
        feature_value = sample[feature_idx]
        threshold = node['threshold']
        
        if feature_value <= threshold:
            return self._traverse_tree(node['left_child'], sample)
        else:
            return self._traverse_tree(node['right_child'], sample)

    def _predict_tree(self, tree: dict, sample: np.ndarray):
        """Get weighted prediction from a single tree."""
        return self._traverse_tree(tree['structure'], sample)

    def predict(self, X: np.ndarray):
        """Predict classes using weighted voting across all trees."""
        if self.trees is None:
            raise ValueError("Forest is not loaded.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_classes = self.metadata['n_classes']
        predictions = []
        
        for sample in X:
            # Accumulate weighted votes from each tree
            class_votes = np.zeros(n_classes)
            
            for tree in self.trees:
                tree_proba = self._predict_tree(tree, sample)
                class_votes += tree_proba
            
            # Predict the class with highest weighted vote
            majority_class = np.argmax(class_votes)
            predictions.append(majority_class)
        
        return np.array(predictions)
        
    def predict_proba(self, X: np.ndarray):
        """Predict class probabilities using average of tree predictions."""
        if self.trees is None:
            raise ValueError("Forest is not loaded.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_classes = self.metadata['n_classes']
        probas = []
        
        for sample in X:
            # Collect predictions from all trees
            tree_predictions = [
                self._predict_tree(tree, sample) 
                for tree in self.trees
            ]
            
            # Calculate probability as proportion of votes
            vote_counts = Counter(tree_predictions)
            sample_proba = np.zeros(n_classes)
            
            for class_idx, count in vote_counts.items():
                sample_proba[class_idx] = count / self.n_estimators
            
            probas.append(sample_proba)
        
        return np.array(probas)
    
    def get_feature_importances(self):
        """Get feature importances if available."""
        if 'feature_importances' in self.metadata:
            return np.array(self.metadata['feature_importances']['importances'])
        return None