import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
import json
import numpy as np

class WarmupCosineRestartLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, restart_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.restart_epochs = restart_epochs
        self.min_lr = min_lr
        super(WarmupCosineRestartLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        current_epoch = self.last_epoch % self.restart_epochs
        if current_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = float(current_epoch) / float(max(1, self.warmup_epochs))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_factor = 0.5 * (1 + math.cos(math.pi * (current_epoch - self.warmup_epochs) / (self.restart_epochs - self.warmup_epochs)))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]

import matplotlib.pyplot as plt
def visualization(list1, list2):
    
    # 确保两个列表长度相等
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度必须相等")
    
    # 创建图形
    plt.figure(figsize=(32, 18))
    
    # 绘制散点图
    plt.scatter(list1, list2, color='blue', label='Data Points')
    
    # 添加标题和标签
    plt.title('二维坐标图')
    plt.xlabel('list1')
    plt.ylabel('list2')
    
    # 显示图例
    plt.legend()
    
    # 显示网格
    plt.grid(True)

    plt.savefig('./visualize.png')



# Example usage
if __name__ == "__main__":
    model = torch.nn.Linear(10, 2)  # Example model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    warmup_epochs = 5
    max_epochs = 100
    restart_epochs = 20

    scheduler = WarmupCosineRestartLR(optimizer, warmup_epochs, max_epochs, restart_epochs)

    lrs = []
    epoches = []
    for epoch in range(max_epochs):
        # Training code would go here
        scheduler.step()
        lrs.append(scheduler.get_lr())
        epoches.append(epoch+1)
        print(f"Epoch {epoch+1}: Learning Rate: {scheduler.get_lr()}")



    visualization(epoches, lrs)
