import numpy as np

def sgd(weights, gradients, learning_rate):
    """
    Stochastic Gradient Descent optimizer.
    
    Args:
        weights: Current weight parameters
        gradients: Gradients of the loss with respect to weights
        learning_rate: Step size for updates
    
    Returns:
        Updated weights
    """
    return weights - learning_rate * gradients


def sgd_momentum(weights, gradients, velocity, learning_rate, momentum=0.9):
    """
    SGD with momentum optimizer.
    
    Args:
        weights: Current weight parameters
        gradients: Gradients of the loss with respect to weights
        velocity: Momentum velocity term
        learning_rate: Step size for updates
        momentum: Momentum coefficient (default: 0.9)
    
    Returns:
        Tuple of (updated weights, updated velocity)
    """
    velocity = momentum * velocity - learning_rate * gradients
    weights = weights + velocity
    return weights, velocity


def adam(weights, gradients, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Args:
        weights: Current weight parameters
        gradients: Gradients of the loss with respect to weights
        m: First moment estimate
        v: Second moment estimate
        t: Time step
        learning_rate: Step size for updates (default: 0.001)
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
    
    Returns:
        Tuple of (updated weights, updated m, updated v)
    """
    m = beta1 * m + (1 - beta1) * gradients
    v = beta2 * v + (1 - beta2) * (gradients ** 2)
    
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return weights, m, v


def rmsprop(weights, gradients, cache, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
    """
    RMSprop optimizer.
    
    Args:
        weights: Current weight parameters
        gradients: Gradients of the loss with respect to weights
        cache: Running average of squared gradients
        learning_rate: Step size for updates (default: 0.001)
        decay_rate: Decay rate for moving average (default: 0.9)
        epsilon: Small constant for numerical stability (default: 1e-8)
    
    Returns:
        Tuple of (updated weights, updated cache)
    """
    cache = decay_rate * cache + (1 - decay_rate) * (gradients ** 2)
    weights = weights - learning_rate * gradients / (np.sqrt(cache) + epsilon)
    
    return weights, cache