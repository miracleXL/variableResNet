import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Lambda
import torchinfo

import numpy as np
import time
from sklearn.metrics import roc_auc_score, roc_curve

class CustomDataset(Dataset):
    def __init__(self, data:np.ndarray, labels:np.ndarray, transform=ToTensor(), 
    target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))):
        self.data:torch.Tensor = torch.from_numpy(data)
        # self.data:torch.Tensor = torch.from_numpy(np.swapaxes(data, 1, 2))
        self.labels:torch.Tensor = torch.from_numpy(labels)
        self.transform = None
        self.target_transform = None
        # self.transform = transform
        # self.target_transform = target_transform
        # self.shuffle()
    
    def shuffle(self, seed=None):
        '\n        seed(self, seed=None)\n\n        Reseed a legacy MT19937 BitGenerator\n        '
        self.shuffle_seed = np.random.randint(1, 65535) if seed is None else seed
        print(f"随机种子：{self.shuffle_seed}")
        np.random.seed(self.shuffle_seed)
        np.random.shuffle(self.data)
        np.random.seed(self.shuffle_seed)
        np.random.shuffle(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx, 0]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

def load_numpy_dataset(path="dataset.npz", train_percent=0.8) -> tuple:
    with np.load(path) as dataset:
        full_data = dataset["data"].astype(np.float32)
        full_labels = dataset["labels"].astype(np.int64)
    train_size = int(full_data.shape[0]*train_percent)
    test_size = full_data.shape[0]-train_size
    seed = np.random.randint(1, 65535) # 35468
    np.random.seed(seed)
    np.random.shuffle(full_data)
    np.random.seed(seed)
    np.random.shuffle(full_labels)
    train_data, test_data = full_data[:train_size], full_data[train_size:]
    train_labels, test_labels = full_labels[:train_size], full_labels[train_size:]
    print(f"训练集大小：{train_size}", f"测试集大小：{test_size}", f"随机种子：{seed}")
    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)
    return train_dataset, test_dataset

def train(
        model:nn.Module,
        loss_fn,
        optimizer:torch.optim.Optimizer,
        dataloader:DataLoader,
        epochs:int = 1,
        device:torch.DeviceObjType = None,
        test_dataloader:DataLoader = None
    ):
    size = len(dataloader.dataset)
    num_batches = size // dataloader.batch_size
    _, (X, y) = next(enumerate(dataloader))
    batch_size = X.shape
    time_collection = []
    loss_collection = []
    correct_collection = []
    cache:list[torch.Tensor] = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        model.train()
        loss, correct = 0, 0

        time_delta = 0
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            # 计时
            torch.cuda.synchronize()
            time_start = time.time()

            # Compute prediction error
            pred = model.forward(X)
            batch_loss = loss_fn(pred, y)

            # Backpropagation
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 计时结束
            torch.cuda.synchronize()
            time_end = time.time()

            current = batch * batch_size + len(X)

            batch_loss = batch_loss.item()
            loss += batch_loss

            batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += batch_correct
            batch_correct /= len(X)

            batch_time = time_end - time_start
            time_delta += batch_time
            print(f"\r{batch+1}/{num_batches+1}  [{current:>3d}/{size:>3d}] - batch loss: {batch_loss:>7f} - batch accuracy: {(100*batch_correct):>0.1f}% - {batch_time*1000:>0.3f}ms", end = "", flush=True)
        loss /= num_batches
        correct /= size
        print(f"\n-- Average loss: {loss:>7f} - Accuracy: {(100*correct):>0.1f}% - {time_delta/num_batches*1000:>0.3f}ms/batch")
        time_collection.append(time_delta)
        loss_collection.append(loss)
        correct_collection.append(correct)
        if test_dataloader is not None:
            mytest(test_dataloader, device=device)
    print("\n", torchinfo.summary(model, input_size=dataloader))
    return correct_collection, loss_collection, time_collection

def test(
        model:nn.Module,
        loss_fn,
        dataloader:DataLoader,
        device:torch.DeviceObjType = None,
        return_preds:bool = False,
        return_roc_auc:bool = False
    ):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    ys = []
    preds = []
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model.forward(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if return_preds:
                ys = np.hstack((ys, y.cpu()))
                preds = np.hstack((preds, pred.argmax(1).cpu()))
    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f} \n")
    if return_roc_auc:
        fpr, tpr, thresholds = roc_curve(ys, preds)
        auc_score = roc_auc_score(ys, preds)
        if return_preds:
            return (ys, preds), (fpr, tpr, thresholds), auc_score
        return (fpr, tpr, thresholds), auc_score
    if return_preds:
        return ys, preds

def mytrain(
        model:nn.Module,
        dataloader:DataLoader,
        epochs:int = 1,
        device:torch.DeviceObjType = None,
        add_layer:function = lambda epoch:False,
        use_cache:bool = False,
        test_dataloader:DataLoader = None
    ):
    size = len(dataloader.dataset)
    num_batches = size // dataloader.batch_size
    _, (X, y) = next(enumerate(dataloader))
    batch_size = X.shape
    time_collection = []
    loss_collection = []
    correct_collection = []
    cache:list[torch.Tensor] = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        model.train()
        if add_layer(epoch):
            model.addNewBlock()
        loss, correct = 0, 0

        time_delta = 0
        for batch, (X, y) in enumerate(dataloader):
            if use_cache:
                if epoch == 0:
                    cache.append(X)
                X = cache[batch].to(device)
            else:
                X = X.to(device)
            y = y.to(device)

            # 计时
            torch.cuda.synchronize()
            time_start = time.time()

            # Compute prediction error
            pred = model.forward(X)
            batch_loss = model.loss_fn(pred, y)

            # Backpropagation
            batch_loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

            # 计时结束
            torch.cuda.synchronize()
            time_end = time.time()

            current = batch * batch_size + len(X)

            batch_loss = batch_loss.item()
            loss += batch_loss

            batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += batch_correct
            batch_correct /= len(X)

            batch_time = time_end - time_start
            time_delta += batch_time
            print(f"\r{batch+1}/{num_batches+1}  [{current:>3d}/{size:>3d}] - batch loss: {batch_loss:>7f} - batch accuracy: {(100*batch_correct):>0.1f}% - {batch_time*1000:>0.3f}ms", end = "", flush=True)
        loss /= num_batches
        correct /= size
        print(f"\n-- Average loss: {loss:>7f} - Accuracy: {(100*correct):>0.1f}% - {time_delta/num_batches*1000:>0.3f}ms/batch")
        time_collection.append(time_delta)
        loss_collection.append(loss)
        correct_collection.append(correct)
        if test_dataloader is not None:
            mytest(test_dataloader, device=device)
    print("\n", torchinfo.summary(model, input_size=dataloader))
    return correct_collection, loss_collection, time_collection

def mytest(
        model:nn.Module,
        dataloader:DataLoader,
        device:torch.DeviceObjType = None,
        return_preds:bool = False,
        return_roc_auc:bool = False
    ):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    ys = []
    preds = []
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model.forward(X)
            test_loss += model.loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if return_preds:
                ys = np.hstack((ys, y.cpu()))
                preds = np.hstack((preds, pred.argmax(1).cpu()))
    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f} \n")
    if return_roc_auc:
        fpr, tpr, thresholds = roc_curve(ys, preds)
        auc_score = roc_auc_score(ys, preds)
        if return_preds:
            return (ys, preds), (fpr, tpr, thresholds), auc_score
        return (fpr, tpr, thresholds), auc_score
    if return_preds:
        return ys, preds