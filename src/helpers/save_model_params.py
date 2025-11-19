import numpy as np

def extract_mlp_to_dict(mlp, feature_names=None):
    """
    Extract sklearn MLPClassifier parameters to a JSON-serializable dictionary.
    
    Parameters:
    -----------
    mlp : sklearn.neural_network.MLPClassifier
        Fitted MLPClassifier model
    feature_names : list, optional
        Names of input features
        
    Returns:
    --------
    dict : Dictionary containing all model parameters and metadata
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(mlp.n_features_in_)]
    
    # Extract weights and biases
    weights = [w.tolist() for w in mlp.coefs_]
    biases = [b.tolist() for b in mlp.intercepts_]
    
    # Build layer information
    layers = []
    layer_sizes = [mlp.n_features_in_] + list(mlp.hidden_layer_sizes) + [mlp.n_outputs_]
    
    for i, (layer_size, weights_matrix, bias_vector) in enumerate(zip(layer_sizes[1:], weights, biases)):
        layer_info = {
            'layer_index': i,
            'input_size': layer_sizes[i],
            'output_size': layer_size,
            'weights': weights_matrix,
            'biases': bias_vector,
            'activation': mlp.activation if i < len(layer_sizes) - 2 else mlp.out_activation_
        }
        layers.append(layer_info)
    
    mlp_dict = {
        'metadata': {
            'model_type': 'MLPClassifier',
            'n_features': int(mlp.n_features_in_),
            'n_classes': int(mlp.n_outputs_),
            'n_layers': int(mlp.n_layers_),
            'n_iter': int(mlp.n_iter_),
            'loss': float(mlp.loss_) if hasattr(mlp, 'loss_') else None,
            'feature_names': feature_names,
            'classes': mlp.classes_.tolist() if hasattr(mlp, 'classes_') else None
        },
        'architecture': {
            'hidden_layer_sizes': list(mlp.hidden_layer_sizes),
            'activation': mlp.activation,
            'output_activation': mlp.out_activation_,
            'solver': mlp.solver,
            'alpha': float(mlp.alpha),
            'batch_size': mlp.batch_size,
            'learning_rate': mlp.learning_rate,
            'learning_rate_init': float(mlp.learning_rate_init),
            'max_iter': int(mlp.max_iter),
            'shuffle': bool(mlp.shuffle),
            'random_state': int(mlp.random_state) if mlp.random_state is not None else None,
            'tol': float(mlp.tol),
            'early_stopping': bool(mlp.early_stopping),
            'validation_fraction': float(mlp.validation_fraction)
        },
        'layers': layers
    }
    
    # Add training history if available
    if hasattr(mlp, 'loss_curve_'):
        mlp_dict['training_history'] = {
            'loss_curve': [float(loss) for loss in mlp.loss_curve_]
        }
        if hasattr(mlp, 'validation_scores_'):
            mlp_dict['training_history']['validation_scores'] = [
                float(score) for score in mlp.validation_scores_
            ]
    
    return mlp_dict

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