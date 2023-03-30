import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional, Callable
import numpy as np

DATA_ROOT = "~/data/cifar-10"

class CIFAR10_Decentralized(CIFAR10):
    def __init__(self, root: str, train: bool = True, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 download: bool = False,
                 index: int = None,
                 weights: List[float] = None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        
        self.__check_index(index=index)
        self.__check_weight(weights=weights)
        
        self.index = index
        self.weights = weights
        
        self.__modify_train_test()
        
    def __modify_train_test(self):
        __len = len(self.data)
        start = int(__len * sum(self.weights[:self.index]))
        stop = int(__len * sum(self.weights[:self.index + 1]))
        self.data = self.data[start : stop]
        self.targets = self.targets[start : stop]
        
    def __check_index(self, index):
        if index == None:
            raise Exception("index cannot be None")
        elif not isinstance(index, int):
            raise Exception(f"index must be an integer but found {type(index)} instead")
    
    def __check_weight(self, weights):
        if weights == None:
            raise Exception("weight cannot be None")
        elif not isinstance(weights, list):
            raise Exception(f"weight must be a list number but found {type(weights)} instead")
        

def load_data(local_num = 3, batch_size = 32) -> List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]]:
    
    dist = np.random.dirichlet(np.ones(local_num), size=1)[0].tolist()
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    data = []
    for idx, d in enumerate(dist):
        trainset = CIFAR10_Decentralized(DATA_ROOT, train=True, download=True, transform=transform, index=idx, weights=dist)
        testset = CIFAR10_Decentralized(DATA_ROOT, train=False, download=True, transform=transform, index=idx, weights=dist)
        
        train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size = batch_size, shuffle=False)
        
        num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
        
        data.append((
            train_loader,
            test_loader,
            num_examples
        ))
    
    return data

if __name__ == "__main__":
    data = load_data(3)
    
    print(len(data))