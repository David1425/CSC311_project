from sklearn.tree import DecisionTreeClassifier

from src.helpers.save_model_params import extract_tree_to_dict

import json

class SklearnDecisionTreeModel:
    def __init__(self, max_depth=None, random_state=None, min_samples_split=2, criterion='gini'):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state,
                                             min_samples_split=min_samples_split, criterion=criterion)

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, json_path):
        tree_dict = extract_tree_to_dict(self.model)

        with open(json_path, 'w') as f:
            json.dump(tree_dict, f, indent=2)

        print(f"Tree saved to: {json_path}")

    def get_depth(self):
        return self.model.get_depth()
    
    def get_n_leaves(self):
        return self.model.get_n_leaves()
    
    def get_params(self):
        return self.model.get_params()
    
    def node_count(self):
        return self.model.tree_.node_count