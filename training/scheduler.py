# training/scheduler.py

class LinearScheduler:
    """
    Linearly decays a value (e.g., epsilon or learning rate) over time.
    """

    def __init__(self, start, end, duration):
        """
        Args:
            start (float): initial value
            end (float): final value
            duration (int): number of steps/epochs to decay over
        """
        self.start = start
        self.end = end
        self.duration = duration
        self.step = 0

    def get_value(self):
        """
        Returns the current value based on decay.
        """
        if self.step >= self.duration:
            return self.end
        ratio = self.step / self.duration
        return self.start + (self.end - self.start) * ratio

    def advance(self):
        """
        Increments the internal step counter.
        """
        self.step += 1


class WarmupDecayScheduler:
    """
    Warm-up followed by exponential decay. Good for learning rates.
    """

    def __init__(self, base_lr, warmup_steps, decay_factor, decay_step_size):
        """
        Args:
            base_lr (float): target learning rate
            warmup_steps (int): steps to warm up from 0 to base_lr
            decay_factor (float): multiply LR by this every decay_step_size
            decay_step_size (int): how often to decay
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.decay_step_size = decay_step_size
        self.step = 0

    def get_lr(self):
        if self.step < self.warmup_steps:
            return self.base_lr * (self.step / self.warmup_steps)
        decay_steps = (self.step - self.warmup_steps) // self.decay_step_size
        return self.base_lr * (self.decay_factor ** decay_steps)

    def advance(self):
        self.step += 1
