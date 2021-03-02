import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from efficientnet_pytorch.model import efficientnet


class LaserScanDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).unsqueeze(1)
        self.y = torch.tensor(y).unsqueeze(1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx]
        y = self.y[idx]

        return x.float(), y.long()


class LaserScanDatasetBranch(Dataset):
    def __init__(self, x, y1, y2):
        self.x = torch.tensor(x).unsqueeze(1)
        self.y1 = torch.tensor(y1).unsqueeze(1)
        self.y2 = torch.tensor(y2).unsqueeze(1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx]
        y1 = self.y1[idx]
        y2 = self.y2[idx]

        return x.float(), y1.long(), y2.long()


class DepthDataset(LaserScanDataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).unsqueeze(1)
        self.y = torch.tensor(y).unsqueeze(1)


class Classifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.l1 = nn.Linear(640, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.fc(x)

        return x


class Regressor(nn.Module):
    def __init__(self, hidden_size):
        super(Regressor, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(640, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.fc(x)

        return x


class CNNClassifier(nn.Module):
    def __init__(self, kernel_size, hidden_size, num_labels):
        super(CNNClassifier, self).__init__()
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        # self.cnn = efficientnet('efficientnet-b0', True, 1, False, self.hidden_size)
        self.fc = nn.Linear(hidden_size * 640, num_labels)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.cnn(x)
        x = self.fc(x.view(x.shape[0], -1))

        return x


class CNNRegressor(nn.Module):
    def __init__(self, kernel_size, hidden_size):
        super(CNNRegressor, self).__init__()
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        # self.cnn = efficientnet('efficientnet-b0', True, 1, False, self.hidden_size)
        self.fc = nn.Linear(hidden_size * 640, 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.cnn(x)
        x = self.fc(x.view(x.shape[0], -1))

        return x


class ClassifierBranch(nn.Module):
    def __init__(self, hidden_size, num_labels1, num_labels2):
        super(ClassifierBranch, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.l1 = nn.Linear(640, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, num_labels1)
        self.fc2 = nn.Linear(hidden_size, num_labels2)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        y1 = self.fc1(x)
        y2 = self.fc2(x)

        return y1, y2


class RegressorBranch(nn.Module):
    def __init__(self, hidden_size):
        super(RegressorBranch, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(640, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        y1 = self.fc1(x)
        y2 = self.fc2(x)

        return y1, y2


class CNNClassifierBranch(nn.Module):
    def __init__(self, kernel_size, hidden_size, num_labels1, num_labels2):
        super(CNNClassifierBranch, self).__init__()
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        # self.cnn = efficientnet('efficientnet-b0', True, 1, False, self.hidden_size)
        self.fc1 = nn.Linear(hidden_size * 640, num_labels1)
        self.fc2 = nn.Linear(hidden_size * 640, num_labels2)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.cnn(x)
        y1 = self.fc1(x.view(x.shape[0], -1))
        y2 = self.fc2(x.view(x.shape[0], -1))

        return y1, y2


class CNNRegressorBranch(nn.Module):
    def __init__(self, kernel_size, hidden_size):
        super(CNNRegressorBranch, self).__init__()
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        # self.cnn = efficientnet('efficientnet-b0', True, 1, False, self.hidden_size)
        self.fc1 = nn.Linear(hidden_size * 640, 1)
        self.fc2 = nn.Linear(hidden_size * 640, 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.cnn(x)
        y1 = self.fc1(x.view(x.shape[0], -1))
        y2 = self.fc2(x.view(x.shape[0], -1))

        return y1, y2
