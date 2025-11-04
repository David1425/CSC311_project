import numpy as np

def extract_tree_to_dict(tree, feature_names=None):
    tree_ = tree.tree_
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree_.n_features)]
    
    def build_json(node_id, depth):
        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]
        feature = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        value = tree_.value[node_id]
        n_samples = tree_.n_node_samples[node_id]
        impurity = tree_.impurity[node_id]
        
        is_leaf = (left_child == right_child)
        
        node_dict = {
            'node_id': int(node_id),
            'depth': int(depth),
            'n_samples': int(n_samples),
            'impurity': float(impurity),
            'is_leaf': bool(is_leaf)
        }
        
        if is_leaf:
            class_counts = value[0].tolist()
            predicted_class = int(np.argmax(value[0]))
            node_dict.update({
                'class_distribution': class_counts,
                'predicted_class': predicted_class
            })
        else:
            node_dict.update({
                'feature': int(feature),
                'feature_name': feature_names[feature],
                'threshold': float(threshold),
                'left': build_json(left_child, depth + 1),
                'right': build_json(right_child, depth + 1)
            })
        
        return node_dict
    
    tree_structure = build_json(0, 0)
    
    tree_dict = {
        'metadata': {
            'n_features': int(tree_.n_features),
            'n_classes': int(tree_.n_classes[0]),
            'n_outputs': int(tree_.n_outputs),
            'max_depth': int(tree_.max_depth),
            'node_count': int(tree_.node_count),
            'feature_names': feature_names
        },
        'tree': tree_structure
    }
    
    return tree_dict