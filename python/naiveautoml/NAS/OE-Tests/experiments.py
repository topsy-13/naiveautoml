import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random


def set_seed(seed=13):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (if available)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables auto-optimization for conv layers (useful for exact reproducibility)

class MLP(nn.Module):
    def __init__(self, input_size, neuron_structure:list, num_classes, hp:dict):
        super().__init__()
        # Define first layer
        layers = []
        prev_size = input_size
        # Create hidden layers
        # Neuron Structure is expecting a list
        for neurons in neuron_structure:
            layers.append(nn.Linear(prev_size, neurons))
            layers.append(nn.ReLU())
            prev_size = neurons

        # Define output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=hp['lr'])
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stopping_patience = 50

    def forward(self, x):
        return self.network(x)
    

    def oe_train(self, train_loader):
        super().train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Flatten images from (batch, 3, 32, 32) to (batch, 3072)
            images = images.view(images.size(0), -1)

            self.optimizer.zero_grad()
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def es_train(self, train_loader, val_loader):
        epoch = 0
        while True:  # Infinite loop until early stopping condition is met
            epoch += 1
            train_loss = self.oe_train(train_loader)
            val_loss, accuracy = self.evaluate(val_loader)

            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                # TODO: Save the model here

            else:
                self.epochs_without_improvement += 1

            # Check for early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs.')
                break
        return train_loss


    def evaluate(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Flatten images
                images = images.view(images.size(0), -1)

                outputs = self(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        self.train()  # Restore training mode
        accuracy = correct / total
        return test_loss / len(test_loader), accuracy
    
    
class Experiment:

    def __init__(self, architecture:dict, train_strategy:str='OE', random_seed:int=13):
        """_summary_

        Args:
        architecture (dict): A dictionary defining the architecture of the model.
            Keys:
                "input_size" (int): The size of the input layer.
                "hlayers_size" (list): A list of integers representing the sizes of the hidden layers, the amount of layers is inferred based on the length of the list.
                "lr" (float): The Learning Rate.
                "num_classes" (int): The len of the last layer

        train_strategy (str, optional): The training strategy to use. Either 'OneEpoch' ('OE') or 'EarlyStopping' ('ES'). Defaults to 'OE'.
        random_seed (int, optional): A seed for random number generation to ensure reproducibility. Defaults to 13.


        Raises:
            ValueError: _description_
        """
        # Get architecture params
        hidden_layers_size = architecture['hlayers_size']
        n_hidden_layers = len(hidden_layers_size)
        lr = architecture['lr']
        

        self.architecture = {
            'input_size': architecture['input_size'],
            'hlayers_size': hidden_layers_size, 
            'n_hlayers': n_hidden_layers, 
            'lr': lr,
            'output_size': architecture['num_classes']

        }
        
        self.strategy = train_strategy
        self.random_seed = random_seed
        
        if train_strategy not in ['OE', 'ES']:
            raise ValueError(f"Invalid architecture '{architecture}'. Valid options are: ['OE', 'ES'].")
        pass
    
            
    
    def build_MLP(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_size=self.architecture['input_size'], 
                         neuron_structure=self.architecture['hlayers_size'],
                         num_classes=self.architecture['output_size'],
                         hp={'lr': self.architecture['lr']}
                         ).to(self.device)

    def build_train_and_evaluate(self, train_loader, test_loader):
        set_seed(self.random_seed)
        self.build_MLP()
        
        # Training strategy
        if self.strategy == 'OE':
            train_loss = self.model.oe_train(train_loader=train_loader)
        elif self.strategy == 'ES':
            train_loss = self.model.es_train(train_loader=train_loader, val_loader=test_loader)

        test_loss, test_accuracy = self.model.evaluate(test_loader)

        # Store results
        results = {
            'n_layers': self.architecture['n_hlayers'],
            'neurons_per_layer': self.architecture['hlayers_size'],
            'learning_rate': self.architecture['lr'],
            'test_accuracy': test_accuracy
        }
        return results


