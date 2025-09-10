from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np

df = pd.read_csv("clicks.csv", header=None)  # 用户浏览行为
df.columns = [
    "session_id",  # 用户ID
    "timestamp",  # 时间戳
    "item_id",  # 浏览商品ID
    "category",  # 商品分类
]
buy_df = pd.read_csv("buys.csv", header=None)  # 用户购买行为
buy_df.columns = [
    "session_id",  # 用户ID
    "timestamp",  # 时间戳
    "item_id",  # 购买商品ID
    "price",  # 总价
    "quantity",  # 数量
]
item_encoder = LabelEncoder()
df["item_id"] = item_encoder.fit_transform(df.item_id)
df.head()
df["label"] = df.session_id.isin(buy_df.session_id)  # 是否购买
df.head()


from torch_geometric.data import InMemoryDataset
from tqdm import tqdm


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):  # 构造函数
        super(YooChooseBinaryDataset, self).__init__(
            root, transform, pre_transform
        )  # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(
        self,
    ):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return []

    @property
    def processed_file_names(
        self,
    ):  # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ["yoochoose_click_binary_1M_sess.dataset"]

    def download(self):
        pass

    def process(self):  # 没有yoochoose_click_binary_1M_sess.dataset文件时，执行该函数

        data_list = []

        # process by session_id
        grouped = df.groupby("session_id")
        for session_id, group in tqdm(
            grouped
        ):  # 遍历每一组session_id（每一个图），目的是将其制作成(from torch_geometric.data import Data)格式
            sess_item_id = LabelEncoder().fit_transform(
                group.item_id
            )  # 将item_id转换成sess_item_id，从0开始。对每一组session_id中的所有item_id进行编码(例如15453,3651,15452)就按照数值大小编码成(2,0,1)，索引是从0开始的
            group = group.reset_index(drop=True)
            group["sess_item_id"] = sess_item_id
            node_features = (
                group.loc[group.session_id == session_id, ["sess_item_id", "item_id"]]
                .sort_values("sess_item_id")
                .item_id.drop_duplicates()
                .values
            )  # item_id作为node_features

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[
                1:
            ]  # 构建邻接矩阵，第二个到最后一个（点的特征就由其ID组成，edge_index是这样，因为咱们浏览的过程中是有顺序的比如(0,0,2,1)- 5.所以边就是0->0,0->2,2->1这样的，对应的索引就为target_nodes: [0 2 1]，source_nodes: [0 0 2]）
            source_nodes = group.sess_item_id.values[:-1]  # 第一个到倒数第二个

            edge_index = torch.tensor(
                [source_nodes, target_nodes], dtype=torch.long
            )  # 传入形成二维邻接矩阵
            x = node_features

            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)  # 传入参数，构建data
            data_list.append(data)  # 每一个图都放入data_list

        data, slices = self.collate(data_list)  # 转换成可以保存到本地的格式
        torch.save((data, slices), self.processed_paths[0])  # 保存数据


dataset = YooChooseBinaryDataset(root="data/")  # 指定保存名字和路径

df = pd.read_csv("clicks.csv", header=None)  # 用户浏览行为
df.columns = [
    "session_id",  # 用户ID
    "timestamp",  # 时间戳
    "item_id",  # 浏览商品ID
    "category",  # 商品分类
]
buy_df = pd.read_csv("buys.csv", header=None)  # 用户购买行为
buy_df.columns = [
    "session_id",  # 用户ID
    "timestamp",  # 时间戳
    "item_id",  # 购买商品ID
    "price",  # 总价
    "quantity",  # 数量
]
item_encoder = LabelEncoder()
df["item_id"] = item_encoder.fit_transform(df.item_id)
df.head()
df["label"] = df.session_id.isin(buy_df.session_id)  # 是否购买
df.head()

# 构建网络结构，一个图做一个分类结果，判断买没买（0/1）
embed_dim = 128
from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class Net(torch.nn.Module):  # 针对图进行分类任务
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(
            embed_dim, 128
        )  # 卷积层，SAGEConv：重构的特征=自己*权重+邻居特征*权重
        self.pool1 = TopKPooling(128, ratio=0.8)  # TopKPooling下采样，比例值
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=df.item_id.max() + 10, embedding_dim=embed_dim
        )  # 每个id做成128维向量
        self.lin1 = torch.nn.Linear(128, 128)  # 全连接
        self.lin2 = torch.nn.Linear(128, 64)  # 全连接
        self.lin3 = torch.nn.Linear(64, 1)  # 全连接
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )  # x:n*1,其中每个图里点的个数是不同的
        # print(x)
        x = self.item_embedding(x)  # n*1*128 特征编码后的结果，每个点做成128维向量
        # print('item_embedding',x.shape)#[212, 1, 128]
        x = x.squeeze(1)  # n*128
        # print('squeeze',x.shape)#[212, 128]
        x = F.relu(self.conv1(x, edge_index))  # 卷积 n*128，传入点和边做更新
        # print('conv1',x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(
            x, edge_index, None, batch
        )  # 池化pool之后得到 n*0.8个点
        # print('self.pool1',x.shape)
        # print('self.pool1',edge_index)
        # print('self.pool1',batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = gap(
            x, batch
        )  # global_mean_pool，全局平均池化，把图中的n个点的128维向量相加再除以n,最后得到1个全局的128维向量
        # print('gmp',gmp(x, batch).shape) # batch*128
        # print('cat',x1.shape) # batch*256
        x = F.relu(self.conv2(x, edge_index))  # 卷积
        # print('conv2',x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)  # 池化
        # print('pool2',x.shape)
        # print('pool2',edge_index)
        # print('pool2',batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = gap(x, batch)
        # print('x2',x2.shape)
        x = F.relu(self.conv3(x, edge_index))  # 卷积
        # print('conv3',x.shape)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)  # 池化
        # print('pool3',x.shape)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        # print('x3',x3.shape)# batch * 256
        x = x1 + x2 + x3  # 获取不同尺度的全局特征，128维

        x = self.lin1(x)  # 全连接
        # print('lin1',x.shape)
        x = self.act1(x)
        x = self.lin2(x)  # 全连接
        # print('lin2',x.shape)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)  # batch个结果，全连接
        # print('sigmoid',x.shape)
        return x


from torch_geometric.loader import DataLoader


def train():  # 训练
    model.train()

    loss_all = 0
    for data in train_loader:  # 遍历
        data = data  # 拿到每个数据
        # print('data',data)
        optimizer.zero_grad()
        output = model(data)  # 传入数据
        label = data.y  # 拿到标签
        loss = crit(output, label)  # 计算损失
        loss.backward()  # 反向传播
        loss_all += data.num_graphs * loss.item()
        optimizer.step()  # 梯度更新
    return loss_all / len(dataset)


model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crit = torch.nn.BCELoss()
train_loader = DataLoader(dataset, batch_size=64)
for epoch in range(10):
    print("epoch:", epoch)
    loss = train()
    print(loss)
# 计算准确率
from sklearn.metrics import roc_auc_score


def evalute(loader, model):
    model.eval()

    prediction = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data  # .to(device)
            pred = model(data)  # .detach().cpu().numpy()

            label = data.y  # .detach().cpu().numpy()
            prediction.append(pred)
            labels.append(label)
    prediction = np.hstack(prediction)
    labels = np.hstack(labels)

    return roc_auc_score(labels, prediction)


for epoch in range(1):
    roc_auc_score = evalute(dataset, model)
    print("roc_auc_score", roc_auc_score)
