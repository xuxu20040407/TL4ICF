"""Original training and transfer learning code for the TL4ICF_DEMO project."""

import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import Linear, PReLU, ReLU
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


class Model(nn.Module):
    def __init__(self, dim_layers):
        super(Model, self).__init__()
        activation_func = PReLU()  # PReLU() or ReLU()
        
        layers = []
        for i in range(len(dim_layers) - 1):
            layers.append((f'layer_{i}', Linear(dim_layers[i], dim_layers[i + 1])))
            if i < len(dim_layers) - 2:  # No activation after the last layer
                layers.append((f'activation_{i}', activation_func))
        
        self.model1 = nn.Sequential(OrderedDict(layers))

    def forward(self, input):
        output = self.model1(input)
        return output


def data_loader(dist, fidelity, tl=False):
    data = np.load(rf".\DATA\{dist}\{fidelity}\data.npy")
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


def main():
    data_distribution = ['random', 'uniform']
    data_fidelity = ['low', 'high', 'exp']
    experiments_TL = ['low2high', 'low2exp', 'high2exp', 'low2high2exp']

    dim_layers = [2, 10, 10, 10, 5, 1]
    num_layers = len(dim_layers) - 1
    frozen_layers = range(num_layers - 2) # Indices of frozen layers
    training_epochs = 1001
    tl_epochs = 5001
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    loss = nn.MSELoss()

    # TODO: Better training function abstraction

    for dist in data_distribution:
        for fidelity in data_fidelity:
            model_path = rf".\models\{dist}_{fidelity}_model.pth"

            # Comment out the following block to train the models again
            if os.path.exists(model_path):
                print(f"Model for {dist} distribution with {fidelity} fidelity already exists.")
                continue

            model = Model(dim_layers).to(device) # Reset model
            print(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            train_dataloader, val_dataloader = data_loader(dist, fidelity)
            
            for epoch in range(training_epochs):
                for batch_idx, (Prob, label) in enumerate(train_dataloader):
                    label, Prob = label.to(device=device), Prob.to(device=device)
                    optimizer.zero_grad()
                    output = model(Prob)
                    result = loss(output, label)
                    result.backward()
                    optimizer.step()
                    # if batch_idx % 5 == 0:
                    #     print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {result}")

                if epoch % 50 == 0:
                    print(f"Epoch: {epoch}, Loss: {result}")
                    for data in val_dataloader:
                        Prob, label = data
                        label, Prob = label.to(device=device), Prob.to(device=device)
                        output = model(Prob)
                        result = loss(output.squeeze(1), label)
                    print(f"Validation Loss: {result}")
                
                # TODO: Save model checkpoint
            
            os.makedirs(rf".\models", exist_ok=True)
            torch.save(model.state_dict(), model_path)

        for experiment in experiments_TL:
            #* Single transfer learning
            model_idxs = {
                'low2high': [0, 1],
                'low2exp': [0, 2],
                'high2exp': [1, 2],
                'low2high2exp': [0, 1]
            }.get(experiment)
            lower_fidelity, higher_fidelity = data_fidelity[model_idxs[0]], data_fidelity[model_idxs[1]]

            model = Model(dim_layers).to(device)
            model.load_state_dict(torch.load(rf".\models\{dist}_{lower_fidelity}_model.pth"))

            # Freeze layers
            for name, param in model.named_parameters():
                param.requires_grad = not any(f'layer_{i}' in name for i in frozen_layers)
                # print(name, param.requires_grad)

            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            train_dataloader, val_dataloader = data_loader(dist, higher_fidelity, tl=True)

            for epoch in range(tl_epochs):
                for data in train_dataloader:
                    Prob, label = data
                    label, Prob = label.to(device=device), Prob.to(device=device)
                    optimizer.zero_grad()
                    output = model(Prob)
                    result = loss(output.squeeze(1), label)
                    result.backward()
                    optimizer.step()
                
                if epoch % 250 == 0:
                    print(f"Epoch: {epoch}, Loss: {result}")
                    for data in val_dataloader:
                        Prob, label = data
                        label, Prob = label.to(device=device), Prob.to(device=device)
                        output = model(Prob)
                        result = loss(output.squeeze(1), label)
                    print(f"Validation Loss: {result}")

            torch.save(model.state_dict(), rf".\models\{dist}_{lower_fidelity}_{higher_fidelity}_model.pth")

            #* Double transfer learning, based on the single transfer learning model
            if experiment == 'low2high2exp':
                model = Model(dim_layers).to(device)
                model.load_state_dict(torch.load(rf".\models\{dist}_low_high_model.pth"))

                for name, param in model.named_parameters():
                    param.requires_grad = not any(f'layer_{i}' in name for i in frozen_layers)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
                train_dataloader, val_dataloader = data_loader(dist, 'exp', tl=True)

                for epoch in range(tl_epochs):
                    for data in train_dataloader:
                        Prob, label = data
                        label, Prob = label.to(device=device), Prob.to(device=device)
                        output = model(Prob)
                        result = loss(output.squeeze(1), label)
                        optimizer.zero_grad()
                        result.backward()
                        optimizer.step()
                    
                    if epoch % 50 == 0:
                        print(f"Epoch: {epoch}, Loss: {result}")
                        for data in val_dataloader:
                            Prob, label = data
                            label, Prob = label.to(device=device), Prob.to(device=device)
                            output = model(Prob)
                            result = loss(output.squeeze(1), label)
                        print(f"Validation Loss: {result}")
                
                torch.save(model.state_dict(), rf".\models\{dist}_low_high_exp_model.pth")


if __name__ == "__main__":
    main()
