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
    
    weights = [w.tolist() for w in mlp.coefs_]
    biases = [b.tolist() for b in mlp.intercepts_]
    
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
    
    if hasattr(mlp, 'loss_curve_'):
        mlp_dict['training_history'] = {
            'loss_curve': [float(loss) for loss in mlp.loss_curve_]
        }
        if hasattr(mlp, 'validation_scores_'):
            mlp_dict['training_history']['validation_scores'] = [
                float(score) for score in mlp.validation_scores_
            ]
    
    return mlp_dict

def extract_tree_to_dict(tree, feature_names=None, classes=None):
    """
    Extract sklearn DecisionTreeClassifier to a JSON-serializable dictionary.
    
    Parameters:
    -----------
    tree : sklearn.tree.DecisionTreeClassifier
        Fitted decision tree model
    feature_names : list, optional
        Names of input features
    classes : array-like, optional
        Class labels for mapping predictions
        
    Returns:
    --------
    dict : Dictionary containing tree structure and metadata
    """
    tree_ = tree.tree_
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree_.n_features)]
    
    if classes is None and hasattr(tree, 'classes_'):
        classes = tree.classes_.tolist()
    
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
    
    if classes is not None:
        tree_dict['metadata']['classes'] = classes if isinstance(classes, list) else classes.tolist()
    
    return tree_dict


def extract_bagging_to_dict(bagging_model, feature_names=None):
    """
    Extract sklearn BaggingClassifier to a JSON-serializable dictionary.
    
    Parameters:
    -----------
    bagging_model : sklearn.ensemble.BaggingClassifier
        Fitted bagging ensemble model
    feature_names : list, optional
        Names of input features
        
    Returns:
    --------
    dict : Dictionary containing all trees and ensemble metadata
    """
    from sklearn.tree import DecisionTreeClassifier
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(bagging_model.n_features_in_)]
    
    classes = bagging_model.classes_.tolist()
    
    trees = []
    estimators_features = []
    
    for idx, estimator in enumerate(bagging_model.estimators_):
        if isinstance(estimator, DecisionTreeClassifier):
            if hasattr(bagging_model, 'estimators_features_'):
                feature_indices = bagging_model.estimators_features_[idx].tolist()
            else:
                feature_indices = list(range(bagging_model.n_features_in_))
            
            tree_feature_names = [feature_names[i] for i in feature_indices]
            
            tree_dict = extract_tree_to_dict(estimator, tree_feature_names, classes)
            trees.append(tree_dict['tree'])
            estimators_features.append(feature_indices)
        else:
            raise ValueError(f"Unsupported estimator type: {type(estimator)}")
    
    ensemble_dict = {
        'metadata': {
            'model_type': 'BaggingClassifier',
            'n_estimators': int(bagging_model.n_estimators),
            'n_features': int(bagging_model.n_features_in_),
            'n_classes': int(len(bagging_model.classes_)),
            'classes': bagging_model.classes_.tolist(),
            'feature_names': feature_names,
            'max_samples': bagging_model.max_samples,
            'max_features': bagging_model.max_features,
            'bootstrap': bool(bagging_model.bootstrap),
            'bootstrap_features': bool(bagging_model.bootstrap_features),
            'oob_score': bool(bagging_model.oob_score),
            'warm_start': bool(bagging_model.warm_start),
            'random_state': int(bagging_model.random_state) if bagging_model.random_state is not None else None
        },
        'trees': trees,
        'estimators_features': estimators_features
    }
    
    if hasattr(bagging_model, 'oob_score_'):
        ensemble_dict['metadata']['oob_score_value'] = float(bagging_model.oob_score_)
    
    return ensemble_dict


def extract_boosting_to_dict(boosting_model, feature_names=None):
    """
    Extract sklearn AdaBoostClassifier to a JSON-serializable dictionary.
    
    Parameters:
    -----------
    boosting_model : sklearn.ensemble.AdaBoostClassifier
        Fitted boosting ensemble model
    feature_names : list, optional
        Names of input features
        
    Returns:
    --------
    dict : Dictionary containing all trees, weights, and ensemble metadata
    """
    from sklearn.tree import DecisionTreeClassifier
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(boosting_model.n_features_in_)]
    
    classes = boosting_model.classes_.tolist()
    
    trees = []
    for estimator in boosting_model.estimators_:
        if isinstance(estimator, DecisionTreeClassifier):
            tree_dict = extract_tree_to_dict(estimator, feature_names, classes)
            trees.append(tree_dict['tree'])
        else:
            raise ValueError(f"Unsupported estimator type: {type(estimator)}")
    
    tree_weights = [float(w) for w in boosting_model.estimator_weights_]
    
    ensemble_dict = {
        'metadata': {
            'model_type': 'AdaBoostClassifier',
            'n_estimators': int(boosting_model.n_estimators),
            'n_features': int(boosting_model.n_features_in_),
            'n_classes': int(len(boosting_model.classes_)),
            'classes': boosting_model.classes_.tolist(),
            'feature_names': feature_names,
            'learning_rate': float(boosting_model.learning_rate),
            'algorithm': boosting_model.algorithm,
            'random_state': int(boosting_model.random_state) if boosting_model.random_state is not None else None
        },
        'trees': trees,
        'tree_weights': tree_weights,
        'estimator_errors': [float(e) for e in boosting_model.estimator_errors_] if hasattr(boosting_model, 'estimator_errors_') else []
    }
    
    return ensemble_dict