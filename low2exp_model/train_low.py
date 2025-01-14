import numpy as np
from torch import nn
from torch.nn import Linear, PReLU, ReLU
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import os
from datetime import datetime
import shutil

# Please modify os.chair to your specific path
current_dir=r'.\low2exp_model'
os.chdir(r'C:\Users\wangz\Desktop\激光核聚变\Transfer_learning_demo')


def RemoveDir(log_dir):
    if os.path.exists(log_dir):
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # 重命名 logs 文件夹
        new_log_dir = f"{log_dir}_{timestamp}"
        shutil.move(log_dir, new_log_dir)
        print(f"Renamed existing log directory to {new_log_dir}")

        # 创建新的 logs 文件夹
    os.makedirs(log_dir)
    print(f"Created new log directory: {log_dir}")


RemoveDir(current_dir+r'\low_logs')


class MYDATA(Dataset):
    def __init__(self, data):
        self.normalized_data, self.min_vals, self.max_vals = self.normalize(data)

    def __len__(self):
        return self.normalized_data.size(0)

    def __getitem__(self, idx):
        features = self.normalized_data[idx, 0:2]
        label = self.normalized_data[idx, 2]
        return features, label

    def normalize(self, data):
        min_vals = data.min(0)
        max_vals = data.max(0)

        normalized_data = (data - min_vals) / (max_vals - min_vals)
        normalized_data = torch.tensor(normalized_data, dtype=torch.float32)
        return normalized_data, min_vals, max_vals

    def inverse_normalize(self, normalized_data):
        # 逆归一化数据
        return normalized_data * (self.max_vals - self.min_vals) + self.min_vals


# 使用自定义数据集类
train_dataset = MYDATA(np.load(r".\DATA\Low\train.npy"))
train_dataloader = DataLoader(train_dataset, batch_size=100, drop_last=True, shuffle
=True)
val_dataset = MYDATA(np.load(r".\DATA\Low\val.npy"))
val_dataloader = DataLoader(val_dataset, batch_size=100, drop_last=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = nn.Sequential(
            Linear(2, 10),
            ReLU(),
            Linear(10, 10),
            ReLU(),
            Linear(10, 10),
            ReLU(),
            Linear(10, 5),
            ReLU(),
            Linear(5, 1),
        )

    def forward(self, input):
        output = self.model1(input)
        return output

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model().to(device)
print(model)

loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

epochs = 1001

writer = SummaryWriter(current_dir+r"\low_logs")

for epoch in range(epochs):
    running_loss = 0.0
    for data in train_dataloader:
        Prob, label = data
        label, Prob = label.to(device=device), Prob.to(device=device)
        output = model(Prob)
        result = loss(output.squeeze(1), label)
        optimizer.zero_grad()
        result.backward()
        optimizer.step()
        running_loss = running_loss + result
    if epoch % 50 == 0:
        print('epoch:' + str(epoch) + '\nrunning_loss=' + str(running_loss))
    writer.add_scalar("running_loss", torch.log(running_loss), epoch)

    test_loss = 0.0
    for data in val_dataloader:
        Prob, label = data
        label, Prob = label.to(device=device), Prob.to(device=device)
        output = model(Prob)
        result = loss(output.squeeze(1), label)
        test_loss = test_loss + result
    if epoch % 50 == 0:
        print('test_loss=' + str(test_loss))
    writer.add_scalar("test_loss", torch.log(test_loss), epoch)
    writer.flush()


torch.save(model.state_dict(), current_dir+r'\low_model.pth')
torch.save(model.state_dict(), r'.\low2high2exp_model\low_model.pth')



