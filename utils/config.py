#use dataclass if >=python3.7
class ParamConfig:
    batch_size: int = 32
    epochs: int = 10
    workers: int = 4
    cuda: bool = True
    use_half: bool = True

    def __init__(self, batch_size=batch_size, epochs=epochs, workers=workers, cuda=cuda, use_half=use_half):
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers
        self.cuda = cuda
        self.use_half = use_half
