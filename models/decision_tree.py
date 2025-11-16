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


class BaggingTreeModel:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.trees = []
        self.estimators_features = []
        self.metadata = None
        self.n_estimators = 0
        self.load_ensemble(json_path)
    
    def load_ensemble(self, path: str) -> None:
        with open(path, 'r') as f:
            ensemble_data = json.load(f)
        
        self.metadata = ensemble_data['metadata']
        self.n_estimators = ensemble_data['metadata']['n_estimators']
        self.trees = ensemble_data['trees']
        
        if 'estimators_features' in ensemble_data:
            self.estimators_features = ensemble_data['estimators_features']
        else:
            n_features = self.metadata['n_features']
            self.estimators_features = [list(range(n_features)) for _ in range(self.n_estimators)]
        
        print(f"Bagging ensemble loaded from: {path}")
        print(f"Number of trees: {self.n_estimators}")
    
    def _traverse_tree(self, node: dict, sample: np.ndarray): 
        if node['is_leaf']:
            return np.array(node['class_distribution'])
        
        feature_idx = node['feature']
        feature_value = sample[feature_idx]
        threshold = node['threshold']
        
        if feature_value <= threshold:
            return self._traverse_tree(node['left'], sample)
        else:
            return self._traverse_tree(node['right'], sample)

    def predict(self, X: np.ndarray):
        if not self.trees:
            raise ValueError("Ensemble is not loaded.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_classes = self.metadata['n_classes']
        classes = np.array(self.metadata['classes'])
        
        avg_probas = np.zeros((len(X), n_classes))
        
        for tree_idx, (tree, feature_indices) in enumerate(zip(self.trees, self.estimators_features)):
            for sample_idx, sample in enumerate(X):
                tree_sample = sample[feature_indices]
                tree_proba = self._traverse_tree(tree, tree_sample)
                avg_probas[sample_idx] += tree_proba
        
        avg_probas /= self.n_estimators
        
        prediction_indices = np.argmax(avg_probas, axis=1)
        
        predictions = classes[prediction_indices]
        
        return predictions

class BoostingTreeModel:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.trees = []
        self.tree_weights = []
        self.metadata = None
        self.n_estimators = 0
        self.load_ensemble(json_path)
    
    def load_ensemble(self, path: str) -> None:
        with open(path, 'r') as f:
            ensemble_data = json.load(f)
        
        self.metadata = ensemble_data['metadata']
        self.n_estimators = ensemble_data['metadata']['n_estimators']
        self.trees = ensemble_data['trees']
        self.tree_weights = ensemble_data['tree_weights']
        print(f"Boosting ensemble loaded from: {path}")
        print(f"Number of trees: {self.n_estimators}")
    
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
        if not self.trees:
            raise ValueError("Ensemble is not loaded.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_classes = self.metadata['n_classes']
        classes = np.array(self.metadata['classes'])
        
        weighted_votes = np.zeros((len(X), n_classes))
        
        for tree_idx, (tree, weight) in enumerate(zip(self.trees, self.tree_weights)):
            for sample_idx, sample in enumerate(X):
                prediction = self._traverse_tree(tree, sample)
                weighted_votes[sample_idx, prediction] += weight
        
        prediction_indices = np.argmax(weighted_votes, axis=1)
        
        predictions = classes[prediction_indices]
        
        return predictions