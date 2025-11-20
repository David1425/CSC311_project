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

def extract_rf_to_dict(rf, feature_names=None):
    """
    Extract sklearn RandomForestClassifier parameters to a JSON-serializable dictionary.
    
    Parameters:
    -----------
    rf : sklearn.ensemble.RandomForestClassifier
        Fitted RandomForestClassifier model
    feature_names : list, optional
        Names of input features
        
    Returns:
    --------
    dict : Dictionary containing all model parameters and tree structures
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(rf.n_features_in_)]
    
    def extract_tree(tree, tree_index):
        """Extract a single decision tree structure."""
        tree_ = tree.tree_
        
        def recurse(node_id):
            """Recursively extract tree nodes."""
            # Check if leaf node
            if tree_.feature[node_id] == -2:  # Leaf node
                return {
                    'node_id': int(node_id),
                    'is_leaf': True,
                    'value': tree_.value[node_id].tolist(),
                    'n_samples': int(tree_.n_node_samples[node_id]),
                    'impurity': float(tree_.impurity[node_id])
                }
            
            # Internal node
            left_child = int(tree_.children_left[node_id])
            right_child = int(tree_.children_right[node_id])
            
            return {
                'node_id': int(node_id),
                'is_leaf': False,
                'feature_index': int(tree_.feature[node_id]),
                'feature_name': feature_names[tree_.feature[node_id]],
                'threshold': float(tree_.threshold[node_id]),
                'impurity': float(tree_.impurity[node_id]),
                'n_samples': int(tree_.n_node_samples[node_id]),
                'value': tree_.value[node_id].tolist(),
                'left_child': recurse(left_child),
                'right_child': recurse(right_child)
            }
        
        return {
            'tree_index': tree_index,
            'n_nodes': int(tree_.node_count),
            'max_depth': int(tree_.max_depth),
            'structure': recurse(0)
        }
    
    # Extract all trees
    trees = [extract_tree(estimator, i) for i, estimator in enumerate(rf.estimators_)]
    
    rf_dict = {
        'metadata': {
            'model_type': 'RandomForestClassifier',
            'n_features': int(rf.n_features_in_),
            'n_classes': int(rf.n_classes_),
            'n_estimators': int(rf.n_estimators),
            'feature_names': feature_names,
            'classes': rf.classes_.tolist() if hasattr(rf, 'classes_') else None
        },
        'hyperparameters': {
            'criterion': rf.criterion,
            'max_depth': int(rf.max_depth) if rf.max_depth is not None else None,
            'min_samples_split': int(rf.min_samples_split) if isinstance(rf.min_samples_split, int) else float(rf.min_samples_split),
            'min_samples_leaf': int(rf.min_samples_leaf) if isinstance(rf.min_samples_leaf, int) else float(rf.min_samples_leaf),
            'min_weight_fraction_leaf': float(rf.min_weight_fraction_leaf),
            'max_features': rf.max_features,
            'max_leaf_nodes': int(rf.max_leaf_nodes) if rf.max_leaf_nodes is not None else None,
            'min_impurity_decrease': float(rf.min_impurity_decrease),
            'bootstrap': bool(rf.bootstrap),
            'oob_score': bool(rf.oob_score),
            'n_jobs': rf.n_jobs,
            'random_state': int(rf.random_state) if rf.random_state is not None else None,
            'max_samples': int(rf.max_samples) if isinstance(rf.max_samples, int) else (float(rf.max_samples) if rf.max_samples is not None else None),
            'ccp_alpha': float(rf.ccp_alpha),
            'class_weight': rf.class_weight if not isinstance(rf.class_weight, dict) else {str(k): float(v) for k, v in rf.class_weight.items()}
        },
        'trees': trees
    }
    
    # Add feature importances
    if hasattr(rf, 'feature_importances_'):
        rf_dict['feature_importances'] = {
            'importances': rf.feature_importances_.tolist(),
            'feature_importance_map': {
                name: float(importance) 
                for name, importance in zip(feature_names, rf.feature_importances_)
            }
        }
    
    # Add OOB score if available
    if hasattr(rf, 'oob_score_'):
        rf_dict['oob_score'] = float(rf.oob_score_)
    
    # Add OOB decision function if available
    if hasattr(rf, 'oob_decision_function_'):
        rf_dict['oob_decision_function'] = rf.oob_decision_function_.tolist()
    
    return rf_dict