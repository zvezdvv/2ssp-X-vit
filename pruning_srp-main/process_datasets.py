from enum import Enum

import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR100 as C100, OxfordIIITPet as OPet
from torchvision.transforms import v2



seed = 42
def set_seed(s):
    global seed; seed = s



class DSSplit(Enum):
    TrainVal = 1
    Test     = 2
class DSName(Enum):
    CIFAR100 = 'cifar100'
    IIIT_PET = 'oxford-iiit-pet'



def seed_gen():
    return torch.Generator().manual_seed(seed)

def load_dataset(
        dataset: DSName, batch_size: int, subset_size: float = 1., res: int = 224, 
        split: DSSplit = DSSplit.Test, download_dataset: bool = False
    ):

    # short aliases
    train, ss, bs = split == DSSplit.TrainVal, subset_size, batch_size
    
    tr = v2.Compose([
        v2.ToImage(),   v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[.5]*3, std=[.5]*3), v2.Resize(res),
    ])
    dataset = (
        C100('data/', train=train, transform=tr, download=download_dataset) \
        if dataset == DSName.CIFAR100 else \
        OPet('data/', split=('trainval' if train else 'test'), transform=tr, download=download_dataset)
    ) 

    if train:
        split = [ss*.9, ss*.1, 1.-ss] if ss < 1. else [.9, .1]
        split = random_split(dataset, split, generator=seed_gen())
        dl = [DataLoader(ds, batch_size=bs, generator=seed_gen()) for ds in split[:2]]
    else:
        if subset_size < 1.: dataset, _ = random_split(dataset, [ss, 1.-ss], generator=seed_gen())
        dl = DataLoader(dataset, batch_size=bs, generator=seed_gen())
    
    return dl