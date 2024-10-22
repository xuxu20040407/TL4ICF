import os
import torch

def train_model(model, train_data, val_data, optimizer, loss_fn, device, 
                epochs=1000, val_interval=50, save_path=None):
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for Prob, label in train_data:
            Prob, label = Prob.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(Prob)
            result = loss_fn(output.squeeze(1), label)
            result.backward()
            optimizer.step()
        
        if epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for Prob, label in val_data:
                    Prob, label = Prob.to(device), label.to(device)
                    output = model(Prob)
                    val_loss += loss_fn(output.squeeze(1), label).item()
                val_loss /= len(val_data)
            print(f"Epoch: {epoch}, Train Loss: {result.item()}, Val Loss: {val_loss}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
