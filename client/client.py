import os, sys
sys.path.append(os.getcwd())
from exp import Train
from data.cifar import load_data
import flwr as fl

import torch
import argparse    
from typing import Dict, List, OrderedDict, Tuple
import numpy as np


class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        trainer: Train,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.trainer = trainer
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.trainer.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.trainer.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        self.trainer.train()
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = self.trainer.test()
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

def main(args) -> None:
    """Load data, start CifarClient."""

    # Load model and data
    trainloader, testloader, num_examples = load_data()[args.cli_idx]
    trainer = Train(args, train_loader=trainloader, test_loader=testloader)

    # Start client
    client = CifarClient(trainer=trainer, trainloader=trainloader, testloader=testloader, num_examples=num_examples)
    fl.client.start_numpy_client(server_address=f"0.0.0.0:{args.port}", client = client)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Federated Learning - Clean Implementation',
                    description='Federated Learning - Clean Implementation - Client',
                    epilog='Enjoy!!! - Author: https://github.com/KhoiDOO')

    parser.add_argument('--port', type=str, default="8080", 
                        help="Server port")  
    parser.add_argument('--cli_idx', type=int, default = 0,
                        help="Client index")  
    parser.add_argument('--epochs', type=int, default=3,
                        help="local round")
    parser.add_argument('--device_index', type=int, default=0,
                        help="index of CUDA device")

    args = parser.parse_args()
    
    main(args=args)