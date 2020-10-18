'''
Model file and non-differentially private file
'''
import time

import torch
import torch.nn.functional as F
from torch import nn, optim

import data
import utils


class EmbeddingNet(nn.Module):
    def __init__(self, vocab_size: int, **_):
        super().__init__()
        # Embedding dimension: vocab_size + <unk>, <pad>, <eos>, <sos>
        self.emb = nn.Embedding(vocab_size + 4, 16)
        self.fc1 = nn.Linear(16, 2)

    def forward(self, x):
        # x: batch_size, seq_len
        x = self.emb(x)  # batch_size, seq_len, embed_dim
        x = x.mean(1)  # batch_size, embed_dim
        x = self.fc1(x)  # batch_size, fc_dim
        return x


class LSTMNet(nn.Module):
    def __init__(self, vocab_size: int, **_):
        super().__init__()
        # Embedding dimension: vocab_size + <unk>, <pad>, <eos>, <sos>
        self.emb = nn.Embedding(vocab_size + 4, 100)
        self.lstm = nn.LSTM(100, 100)
        self.fc1 = nn.Linear(100, 2)

    def forward(self, x):
        # x: batch_size, seq_len
        x = self.emb(x)  # batch_size, seq_len, embed_dim
        x = x.transpose(0, 1)  # seq_len, batch_size, embed_dim
        x, _ = self.lstm(x)  # seq_len, batch_size, lstm_dim
        x = x.mean(0)  # batch_size, lstm_dim
        x = self.fc1(x)  # batch_size, fc_dim
        return x


class MNISTNet(nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x


class FFNN(nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.fc1 = nn.Linear(104, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class Logistic(nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.fc1 = nn.Linear(104, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = F.sigmoid(out)
        return out


class CIFAR10Model(nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.layer_list = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(32, 32, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(32, 64, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 64, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(64, 128, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(128, 256, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Conv2d(256, 10, (3, 3), padding=1, stride=(1, 1)),
        ])

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
            # print(x.shape)
        return torch.mean(x, dim=(2, 3))


model_dict = {
    'mnist': MNISTNet,
    'lstm': LSTMNet,
    'embed': EmbeddingNet,
    'ffnn': FFNN,
    'logreg': Logistic,
    'cifar10': CIFAR10Model,
}


def get_data(args):
    data_fn = data.data_fn_dict[args.experiment][int(args.dummy_data)]
    kwargs = {
        'max_features': args.max_features,
        'max_len': args.max_len,
        'format': 'NCHW',
    }
    if args.dummy_data:
        kwargs['num_examples'] = args.batch_size * 2
    train_data, _ = data_fn(**kwargs)
    for d in train_data:  # train_data, train_labels
        d = torch.from_numpy(d)
        if d.dtype == torch.int32:
            d = d.long()
        if args.experiment == 'logreg' and d.dtype != torch.float32:
            d = d.float()
        yield d


def main(args):
    print(args)
    assert not args.dpsgd
    torch.backends.cudnn.benchmark = True

    train_data, train_labels = get_data(args)
    model = model_dict[args.experiment](vocab_size=args.max_features).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss() if args.experiment != 'logreg' else nn.BCELoss()

    timings = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        dataloader = data.dataloader(train_data, train_labels, args.batch_size)
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            model.zero_grad()
            outputs = model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
        duration = time.perf_counter() - start
        print("Time Taken for Epoch: ", duration)
        timings.append(duration)

    if not args.no_save:
        utils.save_runtimes(__file__.split('.')[0], args, timings)
    else:
        print('Not saving!')
    print('Done!')


if __name__ == '__main__':
    parser = utils.get_parser(model_dict.keys())
    args = parser.parse_args()
    main(args)
