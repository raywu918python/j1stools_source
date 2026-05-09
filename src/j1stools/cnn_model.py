import torch
import torch.nn as nn
import numpy as np

from j1stools.CONFIG import MACDDataBuilterConfig
from j1stools.data_builder import DataBuilder


def prepare_cnn_dataset(features, labels, window_size=20):
    """
    將 DataBuilder 的連續數值轉為 CNN Tensor
    features: d.xtrain (DataFrame)
    labels: d.ytrain (Series/Array)
    """
    # 假設特徵包含 raw_macd, raw_signal, raw_hist
    feature_cols = [col for col in features.columns if "raw_" in col]

    X_list = []
    Y_list = []

    # 這裡假設 DataBuilder 已經幫你處理好視窗切分，如果是平鋪資料則需滑動
    # 若 d.xtrain 已經是 (N, Time, Feat) 的型態則直接轉換
    # 如果還是原始長表格，則執行以下局部縮放邏輯：
    for stock_id, group in features.groupby("stock_id"):
        data = group[feature_cols].values
        target = labels.loc[group.index].values

        for i in range(len(data) - window_size + 1):
            window = data[i : i + window_size].copy()

            # 局部 Min-Max 縮放 (關鍵：讓 CNN 看形狀而非絕對值)
            v_min = window.min(axis=0)
            v_max = window.max(axis=0)
            denom = v_max - v_min
            denom[denom == 0] = 1
            window_norm = (window - v_min) / denom

            X_list.append(window_norm)
            Y_list.append(target[i + window_size - 1])  # 取視窗最後一天的標籤

    X = torch.FloatTensor(np.array(X_list)).transpose(1, 2)  # 轉為 (N, Feat, Time)
    Y = torch.FloatTensor(np.array(Y_list)).view(-1, 1)
    return X, Y


class MacdCNN(nn.Module):
    def __init__(self, num_features=3, window_size=20):
        super(MacdCNN, self).__init__()
        # 第一層卷積：看微觀變動 (Kernel=3)
        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=3, padding=1)
        # 第二層卷積：看中觀趨勢 (Kernel=5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)  # 縮小維度，提取特徵
        self.relu = nn.ReLU()

        # 計算展平後的維度 (經過 Pool 20 -> 10)
        self.fc = nn.Sequential(
            nn.Linear(32 * (window_size // 2), 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 防止過擬合
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 輸出機率
        )

    def forward(self, x):
        # x shape: (Batch, Features, Time)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


cfg = MACDDataBuilterConfig()
cfg.is_continuous = True
d = DataBuilder(cfg).build()

# --- A. 準備資料 ---
window_size = 20
X_train, y_train = prepare_cnn_dataset(d.xtrain, d.ytrain, window_size)

# --- B. 初始化模型 ---
model = MacdCNN(num_features=X_train.shape[1], window_size=window_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# --- C. 訓練循環 (Training Loop) ---
epochs = 100
batch_size = 1024  # 對於 160 萬筆資料，建議分批處理

print("開始訓練 CNN 大腦...")
for epoch in range(epochs):
    model.train()

    # 這裡為練習簡化，直接全量跑。實務上建議使用 torch.utils.data.DataLoader
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        # 計算簡單準確率
        predicted = (outputs > 0.5).float()
        acc = (predicted == y_train).sum() / y_train.shape[0]
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Acc: {acc:.2%}")

print("訓練完成！")
