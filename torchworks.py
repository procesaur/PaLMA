from torch import cuda, device, nn, relu, optim, flatten, cat, sigmoid, load, softmax, max as tmax, from_numpy, no_grad
from math import floor
from torch.utils.data import Dataset, DataLoader
from numpy import array as nparray
from os import path as px


batch = 64
s_len = 128
modelpath = px.join(px.dirname(__file__), "nets/")
devicex = device("cuda:0" if cuda.is_available() else "cpu")


def tensor2device(tensor, print_dev=False):
    tensor = tensor.to(devicex)
    if print_dev:
        print(devicex)
    return tensor


def get_device():
    if cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


class miniNN(nn.Module):

    def __init__(self, num_feature):
        super(miniNN, self).__init__()
        self.layer_out = nn.Linear(num_feature, 2)

    def forward(self, x):
        x = self.layer_out(x)
        return x


class TextClassifier(nn.ModuleList):

    def __init__(self):

        super(TextClassifier, self).__init__()
        self.seq_len: int = 3
        # Model parameters
        self.embedding_size: int = s_len
        self.out_size: int = 32
        self.stride: int = 3
        self.kernels = [4, 5]

        # Training parameters
        self.batch_size: int = batch
        self.learning_rate: float = 0.001
        # Dropout definition
        self.dropout = nn.Dropout(0.25)

        # CNN parameters definition
        self.convs = []
        self.pools = []
        for kernel in self.kernels:
            # Convolution layers definition
            x = nn.Conv1d(self.seq_len, self.out_size, kernel, self.stride).to(devicex)
            self.convs.append(x)
            y = nn.MaxPool1d(kernel, self.stride).to(devicex)
            self.pools.append(y)

        # Fully connected layer definition
        self.fc = nn.Linear(self.in_features_fc(), 2)

    def in_features_fc(self):
        out_pools = []
        for i, kernel in enumerate(self.kernels):
            x = ((self.embedding_size - 1 * (self.kernels[i] - 1) - 1) / self.stride) + 1
            y = ((floor(x) - 1 * (self.kernels[i] - 1) - 1) / self.stride) + 1
            out_pools.append(floor(y))

        # Returns "flattened" vector (input for fully connected layer)
        return (sum(out_pools)) * self.out_size

    def forward(self, x):
        x = x.float()
        outs = []
        for i, k in enumerate(self.kernels):
            y = self.convs[i](x)
            y = relu(y)
            y = self.pools[i](y)
            outs.append(y)

        union = cat(tuple(outs), 2)
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(union)
        # Dropout is applied
        out = self.dropout(out)
        # Activation function is applied
        out = sigmoid(out)

        return out.squeeze()


class DatasetMaper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def pad(seq, target_length=s_len, padding=0):
    length = len(seq)
    if length > target_length:
        seq = seq[:target_length]
    else:
        # seq.extend([padding] * (target_length - length))
        for i in range(target_length-length):
            seq.append(seq[i % length])
    return seq


def netpass(input, net, cnn):
    if cnn:
        model = TextClassifier()
        dataset = DatasetMaper
        input = [[pad(x) for x in y] for y in input]
    else:
        model = miniNN(3)
        dataset = ClassifierDataset

    model.load_state_dict(load(str(px.join(modelpath, net)), map_location=device(devicex)))
    model.to(devicex)
    inputs = nparray(input)
    pseudo_outs = nparray([0 for x in inputs])
    test_dataset = dataset(from_numpy(inputs).float(), from_numpy(pseudo_outs).long())
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    y_pred_list = []
    y_prob_list = []
    with no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(devicex)
            y_test_pred = model(X_batch)
            if cnn:
                y_test_pred = y_test_pred.unsqueeze(0)
            y_pred_softmax = softmax(y_test_pred, dim=1)
            y_pred_prob, y_pred_tags = tmax(y_pred_softmax, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            y_prob_list.append(y_pred_prob.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_prob_list = [a.squeeze().tolist() for a in y_prob_list]
    if cnn:
        y_pred_list[0] = 1-y_pred_list[0]
    return y_pred_list[0], round(y_prob_list[0], 4)
