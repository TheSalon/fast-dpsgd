'''
Opacus experiments for all the models
'''
import time

import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.layers import DPLSTM
from torch import nn, optim

import data
import utils
from pytorch import get_data, model_dict


class LSTMNet(nn.Module):
    def __init__(self, vocab_size: int, batch_size):
        super().__init__()
        # Embedding dimension: vocab_size + <unk>, <pad>, <eos>, <sos>
        self.emb = nn.Embedding(vocab_size + 4, 100)
        self.h_init = torch.randn(1, batch_size, 100).cuda()
        self.c_init = torch.randn(1, batch_size, 100).cuda()
        self.hidden = (self.h_init, self.c_init)
        self.lstm = DPLSTM(100, 100, batch_first=True)
        self.fc1 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.emb(x)  # batch_size, seq_len, embed_dim
        # x has to be of shape [batch_size, seq_len, input_dim]
        x, _ = self.lstm(x, self.hidden)  # batch_size, seq_len, lstm_dim
        x = x.mean(1)  # batch_size, lstm_dim
        x = self.fc1(x)  # batch_size, fc_dim
        return x


def main(args):
    print(args)
    assert args.dpsgd
    torch.backends.cudnn.benchmark = True

    mdict = model_dict.copy()
    mdict['lstm'] = LSTMNet

    train_data, train_labels = get_data(args)
    model = mdict[args.experiment](vocab_size=args.max_features, batch_size=args.batch_size).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0)
    loss_function = nn.CrossEntropyLoss() if args.experiment != 'logreg' else nn.BCELoss()

    privacy_engine = PrivacyEngine(
        model,
        batch_size=args.batch_size,
        sample_size=len(train_data),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=args.sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
    )
    privacy_engine.attach(optimizer)

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
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        print("Time Taken for Epoch: ", duration)
        timings.append(duration)

        if args.dpsgd:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
            print(f"Train Epoch: {epoch} \t"
                  # f"Loss: {np.mean(losses):.6f} "
                  f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}")
        else:
            print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

    if not args.no_save:
        utils.save_runtimes(__file__.split('.')[0], args, timings)
    else:
        print('Not saving!')
    print('Done!')


if __name__ == '__main__':
    parser = utils.get_parser(model_dict.keys())
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Target delta (default: 1e-5)",
    )
    args = parser.parse_args()
    main(args)
