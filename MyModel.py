import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Lambda
import torchinfo

import numpy as np
import time
from sklearn.metrics import roc_auc_score, roc_curve

class CustomDataset(Dataset):
    def __init__(self, data:np.ndarray, labels:np.ndarray, transform:ToTensor=None, target_transform=None):
        self.data:torch.Tensor = torch.from_numpy(data)
        self.labels:torch.Tensor = torch.from_numpy(labels)
        self.transform = None
        self.target_transform = None
        self.transform = transform
        self.target_transform = target_transform
    
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

class MyModelBase(nn.Module):
    # 需要重载build, forward
    def __init__(self, device:torch.device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__built:bool = False
        self.device:torch.device = device
    
    def build(self, input_shape):
        # 待重载
        pass
        self.__built = True
    
    def forward(self, x):
        pass
        return x

    def compile(self, dataloader:DataLoader, loss_fn, optimizer:torch.optim.Optimizer, lr=1e-2):
        self.batch_size:int = dataloader.batch_size
        for X, y in dataloader:
            self.input_shape:tuple = X.shape
            self.output_shape:tuple = y.shape
            break
        self.build(self.input_shape)
        self.loss_fn = loss_fn()
        self.optimizer:torch.optim.Optimizer = optimizer(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        self.to(self.device)
    
    def fit_epoch(self, dataloader:DataLoader, data_size:int, num_batches:int):
        self.train()
        loss, correct = 0, 0

        time_delta = 0
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)

            # 计时
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            time_start = time.time()

            # Compute prediction error
            pred = self.forward(X)
            batch_loss = self.loss_fn(pred, y)

            # Backpropagation
            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 计时结束
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            time_end = time.time()

            current = batch * self.batch_size + len(X)

            batch_loss = batch_loss.item()
            loss += batch_loss

            batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += batch_correct
            batch_correct /= len(X)

            batch_time = time_end - time_start
            time_delta += batch_time
            print(f"\r{batch+1}/{num_batches+1}  [{current:>3d}/{data_size:>3d}] - batch loss: {batch_loss:>7f} - batch accuracy: {(100*batch_correct):>0.1f}% - {batch_time*1000:>0.3f}ms", end = "", flush=True)
        loss /= num_batches
        correct /= data_size
        print(f"\n-- Average loss: {loss:>7f} - Accuracy: {(100*correct):>0.1f}% - {time_delta/num_batches*1000:>0.3f}ms/batch")
        return time_delta, loss, correct

    def fit(self, dataloader:DataLoader, epochs:int=1, test_dataloader=None):
        data_size = len(dataloader.dataset)
        num_batches = data_size // self.batch_size
        time_collection = []
        loss_collection = []
        correct_collection = []
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}/{epochs}")
            time_delta, loss, correct = self.fit_epoch(dataloader, data_size, num_batches)
            time_collection.append(time_delta)
            loss_collection.append(loss)
            correct_collection.append(correct)
            if test_dataloader is not None:
                self.test(test_dataloader)
        print("\n", torchinfo.summary(self, input_size=self.input_shape))
        return correct_collection, loss_collection, time_collection

    def test(self, dataloader:DataLoader, return_preds=False):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        ys = []
        preds = []
        self.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.forward(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                if return_preds:
                    ys = np.hstack((ys, y.cpu()))
                    preds = np.hstack((preds, pred.argmax(1).cpu()))
        test_loss /= num_batches
        correct /= size
        print(f"Test Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f} \n")
        if return_preds:
            return ys, preds
    
    def get_roc_auc(self, dataloader:DataLoader):
        ys, preds = self.test(dataloader, return_preds=True)
        fpr, tpr, thresholds = roc_curve(ys, preds)
        auc_score = roc_auc_score(ys, preds)
        return fpr, tpr, auc_score

class ResBlock(nn.Module):

    def __init__(self, out_channels:int=None, kernel_size:int=3, device=None, *args, **wargs):
        super().__init__(*args, **wargs)
        self.__built = False
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.device = device
        self.return_attn=False

    def build(self, input_shape:tuple):
        self.input_shape = input_shape
        in_L, in_H = input_shape[1], input_shape[2]
        out_channels = in_L if self.out_channels is None else self.out_channels
        self.conv = nn.Conv1d(in_L, out_channels, self.kernel_size, padding=1)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(in_L)
        self.downconv = nn.Conv1d(in_L, out_channels, self.kernel_size, padding=1)
        self.output_shape = input_shape if out_channels == in_L else (input_shape[0], out_channels, *input_shape[2:])
        self.__built = True
        self.to(self.device)

    def forward(self, x:np.ndarray, return_attn=False):
        fx:np.ndarray = x
        fx = self.conv(fx)
        fx = self.bn(fx)
        if fx.shape[-1] != x.shape[-1]:
            x = self.downconv(x)
        return fx + x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

class MyResNet(MyModelBase):
    # 需要重载build, addNewBlock, add_condition
    def __init__(self, device:torch.device=None, copy_block:bool=False, cache:bool=False, freeze_block:bool=False, *args, **kwargs):
        super().__init__(device=device, *args, **kwargs)
        self.__blocks_num:int = 1
        self.__frozen_blocks_num:int = 0
        self.blocks:nn.ModuleList = nn.ModuleList([])
        self.__built:bool = False
        self.cache:bool = cache
        self.__cache:list[torch.TensorType] = []
        self.copy_block:bool = copy_block
        self.freeze_block= freeze_block
        if self.cache:
            self.freeze_block = True
    
    def forward(self, x):
        pass
        return x
    
    def freeze(self, layer:nn.Module):
        if self.freeze_block:
            for param in layer.parameters():
                param.requires_grad = False
    
    def __forward_cache(self, block):
        if self.cache:
            with torch.no_grad():
                for batch, X in enumerate(self.__cache):
                    self.__cache[batch] = block(X.to(self.device))

    def addNewBlock(self, copy_last=False):
        pass
        # print("----------")
        # print(f"{'Copying' if copy_last else 'Adding'} new block...")
        # newBlock = ResBlock()
        # last_block:ResBlock = self.blocks[-1]
        # newBlock.build(last_block.output_shape)
        # if copy_last:
        #     # 复制上一层参数
        #     newBlock.load_state_dict(last_block.state_dict())
        # self.blocks.append(newBlock)
        # newBlock.to(self.device)
        # self.__blocks_num += 1
        # if self.freeze_block:
        #     self.freeze(last_block)
        # self.__frozen_blocks_num += 1
        # self.__forward_cache(last_block)
        # print("Success!")

    def add_condition(self, epoch):
        pass
        # if self.__blocks_num < 4:
        #     if epoch and epoch%5 == 0:
        #         return True
        # return False

    def fit(self, dataloader:DataLoader, epochs:int=1, test_dataloader=None):
        size = len(dataloader.dataset)
        num_batches = size // self.batch_size
        time_collection = []
        loss_collection = []
        correct_collection = []
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}/{epochs}")
            self.train()
            if self.add_condition(epoch):
                self.addNewBlock(self.copy_block)
            loss, correct = 0, 0

            time_delta = 0
            for batch, (X, y) in enumerate(dataloader):
                if self.cache:
                    if epoch == 0:
                        self.__cache.append(X)
                    X = self.__cache[batch].to(self.device)
                else:
                    X = X.to(self.device)
                y = y.to(self.device)

                # 计时
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                time_start = time.time()

                # Compute prediction error
                pred = self.forward(X)
                batch_loss = self.loss_fn(pred, y)

                # Backpropagation
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 计时结束
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                time_end = time.time()

                current = batch * self.batch_size + len(X)

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
                self.test(test_dataloader)
        print("\n", torchinfo.summary(self, input_size=self.input_shape))
        return correct_collection, loss_collection, time_collection
