from torch.utils.tensorboard import SummaryWriter


class NoOpSummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        pass  # Overriding to do nothing

    def add_scalar(self, *args, **kwargs):
        pass  # Overriding to do nothing

    def add_text(self, *args, **kwargs):
        pass  # Overriding to do nothing

    def close(self, *args, **kwargs):
        pass  # Overriding to do nothing
