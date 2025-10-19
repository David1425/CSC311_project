import numpy as np

def constant_lr(initial_lr):
    """Constant learning rate schedule."""
    def schedule(current_step, total_steps):
        return initial_lr
    return schedule


def step_decay(initial_lr, decay_factor=0.5, steps_drop=1000):
    """Step decay learning rate schedule.
    Learning rate is multiplied by decay_factor every fixed number of steps (steps_drop).

    Args:
        initial_lr: Initial learning rate
        decay_factor: Factor by which to reduce learning rate at each drop
        steps_drop: Number of steps between drops
    """
    def schedule(current_step, total_steps):
        return initial_lr * (decay_factor ** (current_step // steps_drop))
    return schedule


def exponential_decay(initial_lr, decay_factor=0.95):
    """Exponential decay learning rate schedule.
    Learning rate is multiplied by decay_factor every step.

    Args:
        initial_lr: Initial learning rate
        decay_factor: Decay factor per step
    """
    def schedule(current_step, total_steps):
        return initial_lr * (decay_factor ** current_step)
    return schedule


def cosine_annealing(initial_lr, end_lr=0):
    """Cosine annealing learning rate schedule.
    
    Args:
        initial_lr: Initial learning rate
        end_lr: Ending learning rate
    """
    def schedule(current_step, total_steps):
        return end_lr + (initial_lr - end_lr) * (1 + np.cos(np.pi * current_step / total_steps)) / 2
    return schedule


def polynomial_decay(initial_lr, power=2.0, end_lr=0):
    """Polynomial decay learning rate schedule.
    Learning rate is equal to initial_lr * (1 - step/total_steps)^power + end_lr.
    
    Args:
        initial_lr: Initial learning rate
        power: Polynomial power
        end_lr: Ending learning rate
    """
    def schedule(current_step, total_steps):
        decay = (1 - current_step / total_steps) ** power
        return (initial_lr - end_lr) * decay + end_lr
    return schedule


def warmup_cosine(initial_lr, warmup_steps=500, end_lr=0):
    """Learning rate schedule with linear warmup followed by cosine annealing.
    Learning rate increases linearly from 0 to initial_lr over warmup_steps,
    then follows a cosine annealing schedule down to end_lr.
    
    Args:
        initial_lr: Initial learning rate when warmup ends
        warmup_steps: Number of warmup steps
        end_lr: Ending learning rate
    """
    def schedule(current_step, total_steps):
        if current_step < warmup_steps:
            # Linear warmup
            return initial_lr * (current_step + 1) / warmup_steps
        else:
            # Cosine annealing
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return end_lr + (initial_lr - end_lr) * (1 + np.cos(np.pi * progress)) / 2
    return schedule