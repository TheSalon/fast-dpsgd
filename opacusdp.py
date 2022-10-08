'''
Opacus experiments for all the models
'''
import time

import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.layers import DPLSTM
from opacus.optimizers import DPOptimizer
from torch import nn, optim

import data
import utils
from pytorch import get_data, model_dict


# Same model as for PyTorch, with DPLSTM replacing nn.LSTM
class LSTMNet(nn.Module):
    def __init__(self, vocab_size: int, **_):
        super().__init__()
        # Embedding dimension: vocab_size + <unk>, <pad>, <eos>, <sos>
        self.emb = nn.Embedding(vocab_size + 4, 100)
        self.lstm = DPLSTM(100, 100)
        self.fc1 = nn.Linear(100, 2)

    def forward(self, x):
        # x: batch_size, seq_len
        x = self.emb(x)  # batch_size, seq_len, embed_dim
        x = x.transpose(0, 1)  # seq_len, batch_size, embed_dim
        x, _ = self.lstm(x)  # seq_len, batch_size, lstm_dim
        x = x.mean(0)  # batch_size, lstm_dim
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

    # Initialize the privacy accountant
    accountant = RDPAccountant()

    # Wrap the model to support per-sample gradients
    model = GradSampleModule(model)

    # Wrap the optimizer to support noise and clipping
    optimizer = DPOptimizer(
        optimizer=optimizer,
        noise_multiplier=args.sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
        expected_batch_size=args.batch_size
    )

    # Attach the accountant to track privacy for the optimizer
    optimizer.attach_step_hook(
        accountant.get_optimizer_hook_fn(
            sample_rate=args.batch_size/len(train_labels)
        )
    )

    timings = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        dataloader = data.dataloader(train_data, train_labels, args.batch_size)
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        print("Time Taken for Epoch: ", duration)
        timings.append(duration)

        epsilon = accountant.get_epsilon(args.delta)
        print(f"Train Epoch: {epoch} \t"
              f"(ε = {epsilon:.2f}, δ = {args.delta})")

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
