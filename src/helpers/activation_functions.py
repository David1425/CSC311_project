import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    
    Args:
        z: Input array or scalar
        
    Returns:
        Sigmoid of z
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid function.
    
    Args:
        z: Input array or scalar
        
    Returns:
        Derivative of sigmoid at z
    """
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Args:
        z: Input array or scalar
        
    Returns:
        ReLU of z
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU function.
    
    Args:
        z: Input array or scalar
        
    Returns:
        Derivative of ReLU at z
    """
    return (z > 0).astype(float)


def tanh(z):
    """
    Hyperbolic tangent activation function.
    
    Args:
        z: Input array or scalar
        
    Returns:
        Tanh of z
    """
    return np.tanh(z)


def tanh_derivative(z):
    """
    Derivative of tanh function.
    
    Args:
        z: Input array or scalar
        
    Returns:
        Derivative of tanh at z
    """
    return 1 - np.tanh(z) ** 2


def softmax(z):
    """
    Softmax activation function.
    
    Args:
        z: Input array (typically 2D: batch_size x num_classes)
        
    Returns:
        Softmax probabilities
    """
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU activation function.
    
    Args:
        z: Input array or scalar
        alpha: Slope for negative values
        
    Returns:
        Leaky ReLU of z
    """
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    """
    Derivative of Leaky ReLU function.
    
    Args:
        z: Input array or scalar
        alpha: Slope for negative values
        
    Returns:
        Derivative of Leaky ReLU at z
    """
    return np.where(z > 0, 1, alpha)