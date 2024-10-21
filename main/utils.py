import yaml
import argparse

import numpy as np
import torch
from torch.nn import PReLU, ReLU
from torch.utils.data import DataLoader, Dataset, random_split


class MYDATA(Dataset):
    def __init__(self, data):
        self.normalized_data, self.min_vals, self.max_vals = self.normalize(data)

    def __len__(self):
        return self.normalized_data.size(0)

    def __getitem__(self, idx):
        Prob = self.normalized_data[idx, 0:2]
        label = self.normalized_data[idx, 2]
        return Prob, label

    def normalize(self, data):
        min_vals = data.min(0)
        max_vals = data.max(0)

        normalized_data = (data - min_vals) / (max_vals - min_vals)
        normalized_data = torch.tensor(normalized_data, dtype=torch.float32)
        return normalized_data, min_vals, max_vals

    def inverse_normalize(self, normalized_data): 
        return normalized_data * (self.max_vals - self.min_vals) + self.min_vals


def data_loader(dist, fidelity, tl=False):
    """ Loads data from a specified distribution and fidelity, optionally applying transfer learning.

    Args:
        dist (str): The distribution type of the data.
        fidelity (str): The fidelity level of the data.
        tl (bool, optional): If True, apply transfer learning by using only the first 5% of the data. Defaults to False.

    Returns:
        tuple: A tuple containing the training and validation DataLoader objects.
    """
    data = np.load(rf".\data\{dist}\{fidelity}\data.npy")
    if tl:
        data = data[:int(0.05 * data.shape[0]), :]  # Take only the first 5% of the data along the first dimension
    dataset = MYDATA(data)
    
    # Split the dataset into training and validation sets (80:20 ratio)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    if tl:
        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=64, drop_last=True, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, drop_last=True)
    return train_dataloader, val_dataloader


def prepare_args(config_file='mlp'):
    """ Prepares the arguments for the training process from a YAML configuration file.
    
    Args:
        config_file (str): Path to the YAML configuration file. Defaults to 'config.yaml'.
    
    Returns:
        argparse.Namespace: An object containing the configuration parameters.
    """
    parser = argparse.ArgumentParser(description='DEMO for Transfer Learning for ICF')
    
    with open(rf'.\configs\{config_file}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    parser.add_argument('--config', type=str, default=config_file, help='Configuration file to use')
    parser.add_argument('--dist', type=str, default=config.get('dist', 'random'), help='Distribution type of the data')
    parser.add_argument('--fidelity', type=str, default=config.get('fidelity', 'low'), help='Fidelity level of the data')
    parser.add_argument('--model', type=str, default=config.get('model', 'MLP'), help='Model type to use for training')
    parser.add_argument('--dim_layers', type=int, nargs='+', default=config.get('dim_layers', [2, 10, 10, 10, 5, 1]), help='Dimensions of the layers in the model')
    parser.add_argument('--activation', type=str, default=config.get('activation', 'PReLU'), help='Activation function to use in the model, PReLU or ReLU')
    parser.add_argument('--optimizer', type=str, default=config.get('optimizer', 'Adam'), help='Optimizer to use: Adam, SGD')
    parser.add_argument('--loss_fn', type=str, default=config.get('loss_fn', 'MSE'), help='Loss function to use: MSE')
    parser.add_argument('--epochs', type=int, default=config.get('epochs', 1001), help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 64), help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.get('lr', 1e-3), help='Learning rate for the optimizer')
    parser.add_argument('--val_interval', type=int, default=config.get('val_interval', 50), help='Validation interval in terms of epochs')
    parser.add_argument('--save_path', type=str, default=config.get('save_path', './models/'), help='Path to save trained model')
    parser.add_argument('--load_path', type=str, default=config.get('load_path', './models/'), help='Path to load pre-trained model for transfer learning')
    parser.add_argument('--seed', type=int, default=config.get('seed', 42), help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=config.get('device', 'cuda'), help='Device to use for training: cuda or cpu')
    parser.add_argument('--mode', type=str, default=config.get('mode', 'train'), help='Mode to run the script: train or tl')

    args = parser.parse_args()
    return args


def get_activation_func(activation):
    if activation == 'PReLU':
        activation_func = PReLU()
    elif activation == 'ReLU':
        activation_func = ReLU()
    else:
        raise ValueError("Invalid activation function. Please choose either 'PReLU' or 'ReLU'.")
    return activation_func


def get_optimizer(optimizer, model, lr):
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer. Please choose either 'Adam' or 'SGD'.")
    return optimizer


def get_loss_fn(loss_fn):
    if loss_fn == 'MSE':
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError("Invalid loss function. Please choose 'MSE'.")
    return loss_fn


def get_model_name(args):
    model_name = f"{args.model}_{'_'.join(map(str, args.dim_layers))}_{args.activation}_{args.optimizer}_{args.loss_fn}"
    return model_name
