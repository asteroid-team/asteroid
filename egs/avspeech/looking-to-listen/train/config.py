# use dataclass if >=python3.7
class ParamConfig:
    def __init__(self, batch_size, epochs, workers, cuda, use_half, learning_rate):
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers
        self.cuda = cuda
        self.use_half = use_half
        self.learning_rate = learning_rate
