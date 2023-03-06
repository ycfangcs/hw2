import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # 看图，搭积木一样拼起来
    modules = nn.Sequential(
        # input_dim和hidden_dim的要求在ipynb有，按要求实现
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),

        nn.Linear(hidden_dim, dim),   
        norm(dim)
    )
    return nn.Sequential(
        nn.Residual(modules),
        # ReLU在Reisual的后面
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    for i in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    correct, loss_sum, n_step, n_samples= 0., 0., 0, 0
    if opt:
        # 训练
        model.train()
    else:
        # 测试
        model.eval()

    # dataloader是迭代器，用迭代的方式取数据
    for X, y in dataloader:
        # 梯度归零
        if opt:
            opt.reset_grad()
        
        # 前向计算，直到算出loss
        pred = model(X)
        loss = loss_func(pred, y)

        # 统计一下预测正确的数目
        correct += (pred.numpy().argmax(axis=1) == y.numpy()).sum()

        # 对loss反向传播梯度（训练时）
        if opt:
            loss.backward()
            opt.step() # 用我们刚实现的优化器里的step函数更新模型参数
        loss_sum += loss.numpy()
        n_step += 1
        n_samples += X.shape[0]
    
    # 返回错误率和平均loss
    return (1 - correct / n_samples), loss_sum / n_step
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # 在MNIST上训练的整体流程

    # 先读数据集
    train_data = ndl.data.MNISTDataset(
        data_dir + '/train-images-idx3-ubyte.gz',
        data_dir + '/train-labels-idx1-ubyte.gz'
    )
    test_data = ndl.data.MNISTDataset(
        data_dir + '/t10k-images-idx3-ubyte.gz',
        data_dir + '/t10k-labels-idx1-ubyte.gz',
    )
    train_loader = ndl.data.DataLoader(train_data, batch_size)
    test_loader = ndl.data.DataLoader(test_data, batch_size)

    # 用上面的模块搭模型
    model = MLPResNet(28 * 28, hidden_dim)

    # 优化器
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 用上面的函数训练
    for _ in range(epochs):
        # 训练一个epoch
        train_acc, train_loss = epoch(train_loader, model, opt)
        # 每训练一个epoch，测试一下
        test_acc, test_loss = epoch(test_loader, model)

    return (train_acc, train_loss, test_acc, test_loss)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
