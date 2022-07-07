# Pytorch tutorial

## 1. 流程

### 1.1 数据

load data

**torch.utils.data.Dataset**

**torch.utils.data.Dataloader**



两种数据集：

map式

这一种必须要重写**getitem**(self, index),**len**(self) 两个内建方法。

目的是当使用dataset[idx]命令时，可以在你的硬盘中读取你的数据集中第idx张图片以及其标签（如果有的话）;len(dataset)则会返回这个数据集的容量。

```python
class CustomDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0
```

iterable式（还没学）

 

之后要用dataloader迭代：

```python
# 创建Dateset(可以自定义)
    dataset = face_dataset # Dataset部分自定义过的face_dataset
# Dataset传递给DataLoader
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=False,num_workers=8)
# DataLoader迭代产生训练数据提供给模型
    for i in range(epoch):
        for index,(img,label) in enumerate(dataloader):
            pass
```

### 1.2 网络 

define neural network

loss function

optimizer

**有torch.nn和torch.optim可以用**

### 1.3 训练

training

vailidation



## 2. 怎么写dataset

```python
class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None): #这里建立数据集的时候直接就要输入x，y；很多例子是输入文件名从文件里读取
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)#简单处理了一下数据格式

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
```

dataset需要写三个函数：

init：一般就预处理一下数据

getitem：根据索引取数据

len：返回data的长度（个数）

## 3. 怎么写model

```python
class My_Model(nn.Module): #简单搭个网络
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B) 这里的B应该是batch size
        return x
```

先继承nn.Moudule

然后写两个函数：

init：定义网络结构

forward：就是把数据丢进网络，通常调用的时候model（x）就是调用了forward

## 4. 怎么写trainer

一般来说会把训练过程写成一个trainer函数

在此函数里：

要指定损失函数criterrion

```python
criterion = nn.MSELoss(reduction='mean') 
```

要指定optimizer

```python
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
```

要写for epoch的循环

```python
for epoch in range(n_epochs):
```

关于epoch的循环：

大致流程是这样的：

```python
model.train()
```

首先设置成训练模式（和验证区分）

然后可以引入一个进度条：

```python
train_pbar = tqdm(train_loader, position=0, leave=True)
# tqdm是用来生成进度条的工具，参数为一个可迭代对象，在迭代他的时候会自动生成进度条。

for x, y in train_pbar:# 循环中迭代x,y
```

然后在循环内进行一系列固定的操作：

```python
optimizer.zero_grad()  # Set gradient to zero.
x, y = x.to(device), y.to(device)  # Move your data to device.
pred = model(x)   #直接用model就是执行forward
loss = criterion(pred, y)  #这里算mse
loss.backward()  # Compute gradient(backpropagation).
optimizer.step()  # Update parameters，就是执行一步backward
step += 1
loss_record.append(loss.detach().item())
```

然后为了使得进度条清楚还可以加一些描述：

```python
train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')#第几个epoch
train_pbar.set_postfix({'loss': loss.detach().item()})
```

然后train就写完了

然后要写val：

```python
model.eval()
```

val并不需要算梯度，只需要算结果和算损失，所以循环会比较不同：

```python
loss_record = []
for x, y in valid_loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred, y)

    loss_record.append(loss.item())
```

取一个平均的loss代表这个val的loss 

```python
mean_valid_loss = sum(loss_record) / len(loss_record)
```

然后打印一下：

```python
print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
```

然后万一模型训练下降了怎么办呢？

我们保存loss最低的模型，同时当一段时间没有提升之后就停止训练

```python
if mean_valid_loss < best_loss:
    best_loss = mean_valid_loss
    torch.save(model.state_dict(), config['save_path'])  # Save your best model
    print('Saving model with loss {:.3f}...'.format(best_loss))
    early_stop_count = 0
else:
    early_stop_count += 1

if early_stop_count >= config['early_stop']:#如果early_stop步都没有提升就停
    print('\nModel is not improving, so we halt the training session.')
    return
```

注意这里教了你怎么保存模型。



## 5. 怎么写main函数

先指定一系列参数：

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'select_all': True,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 3000,  # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
# Set seed for reproducibility
same_seed(config['seed'])
```

之后我们要读取数据：

```python
train_data, test_data = pd.read_csv('./data/covid.train.csv').values, pd.read_csv('./data/covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])
```

读完数据可以打印看一下大小

```python
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")
```

这里使用了一个select_feat的函数将训练集和验证集划分开了：

```python
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])
```

然后搭建了一下dataset

```python
train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \COVID19Dataset(x_valid, y_valid), \COVID19Dataset(x_test)
```

然后用dataloader生成迭代器，然后训练

```python
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

model = My_Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)
```



## 6. 怎么用保存的模型check









