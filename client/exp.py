import os, sys
sys.path.append(os.getcwd())
from backbone.core import Net

from typing import Tuple, Dict

import torch
from torch import nn

class Train:
    def __init__(self, args, train_loader, test_loader) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 
                                   index = args.device_index)
        self.net = Net().to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = args.epochs
        
    
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        print(f"Training {self.epochs} epoch(s) w/ {len(self.train_loader)} batches each")

        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)

                optimizer.zero_grad()

                outputs = self.net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99: 
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0


    def test(self) -> Tuple[float, float]:
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy