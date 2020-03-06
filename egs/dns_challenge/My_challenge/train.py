from asteroid.data.noisy_dataset import NoisyDataSet
from torch.utils.data import DataLoader
import soundfile as sf

batch_size = 1

train = NoisyDataSet()
trainset = DataLoader(train, batch_size=batch_size, shuffle=False)


