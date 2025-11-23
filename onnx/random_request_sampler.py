from torch.utils.data import DataLoader, RandomSampler

class RandomRequestSampler:
    def __init__(self, dataset, batch_size=1):
        sampler = RandomSampler(dataset, replacement=True)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        self.loader_iter = iter(self.data_loader)

    def sample(self):
        try:
            data, label = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.data_loader)
            data, label = next(self.loader_iter)
        return (data, label)
