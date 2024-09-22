import torch
from torch.utils.data import Sampler


class CustomSampler(Sampler):
    def __init__(self, data_source, num_replicas, rank, special_rank=None, special_ratio=2):
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.special_rank = special_rank
        self.special_ratio = special_ratio

        # Calculate the number of samples for each rank
        if rank == special_rank:
            self.num_samples = len(self.data_source) * special_ratio // (num_replicas + special_ratio - 1)
        else:
            self.num_samples = len(self.data_source) // (num_replicas + special_ratio - 1)
            
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.rank == self.special_rank:
            indices = indices * self.special_ratio

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)

    def __len__(self):
        return self.num_samples
