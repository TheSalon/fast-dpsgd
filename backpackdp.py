'''
BackPACK experiments in this file
'''
import time

import torch
import torch.nn.functional as F
from backpack import backpack, extend
from backpack.extensions import BatchGrad, BatchL2Grad
from torch import nn
from torch.optim import Optimizer

import data
import utils
from pytorch import get_data, model_dict


def make_broadcastable(v, X):
    """Returns a view of `v` that can be broadcast with `X`.

    If `v` is a one-dimensional tensor [N] and `X` is a tensor of shape
    `[N, ..., ]`, returns a view of v with singleton dimensions appended.

    Example:
        `v` is a tensor of shape `[10]` and `X` is a tensor of shape `[10, 3, 3]`.
        We want to multiply each `[3, 3]` element of `X` by the corresponding
        element of `v` to get a matrix `Y` of shape `[10, 3, 3]` such that
        `Y[i, a, b] = v[i] * X[i, a, b]`.

        `w = make_broadcastable(v, X)` gives a `w` of shape `[10, 1, 1]`,
        and we can now broadcast `Y = w * X`.
    """
    broadcasting_shape = (-1, *[1 for _ in X.shape[1:]])
    return v.reshape(broadcasting_shape)


class DP_SGD(Optimizer):
    """Differentially Private SGD.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        max_norm (float, optional): maximum norm of the individual gradient,
            to which they will be clipped if exceeded (default: 0.01)
        stddev (float, optional): standard deviation of the added noise
            (default: 1.0)
    """
    def __init__(self, params, lr=0.1, max_norm=1.0, stddev=1.0):
        self.lr = lr
        self.max_norm = max_norm
        self.stddev = stddev
        super().__init__(params, dict())

    def step(self):
        """Performs a single optimization step.

        The function expects the gradients to have been computed by BackPACK
        and the parameters to have a ``batch_l2`` and ``grad_batch`` attribute.
        """
        l2_norms_all_params_list = []
        for group in self.param_groups:
            for p in group["params"]:
                l2_norms_all_params_list.append(p.batch_l2)

        l2_norms_all_params = torch.stack(l2_norms_all_params_list)
        total_norms = torch.sqrt(torch.sum(l2_norms_all_params, dim=0))
        scaling_factors = torch.clamp_max(total_norms / self.max_norm, 1.0)

        for group in self.param_groups:
            for p in group["params"]:
                clipped_grads = p.grad_batch * make_broadcastable(scaling_factors, p.grad_batch)
                clipped_grad = torch.sum(clipped_grads, dim=0)

                noise_magnitude = self.stddev * self.max_norm
                noise = torch.randn_like(clipped_grad) * noise_magnitude

                perturbed_update = clipped_grad + noise

                p.data.add_(-self.lr * perturbed_update)


dpsgd_kwargs = {
    'mnist': {
        'max_norm': 0.01,
        'stddev': 2.0
    },
    # 'lstm': {'max_norm': 1.0, 'stddev': 1.1},
    # 'embed': {'max_norm': 1.0, 'stddev': 1.1},
    'ffnn': {
        'max_norm': 1.0,
        'stddev': 1.1
    },
    'logreg': {
        'max_norm': 1.0,
        'stddev': 1.1
    },
    'cifar10': {
        'max_norm': 1.0,
        'stddev': 1.1
    },
}


def main(args):
    print(args)
    assert args.dpsgd
    torch.backends.cudnn.benchmark = True

    train_data, train_labels = get_data(args)
    model = model_dict[args.experiment](vocab_size=args.max_features).cuda()
    model = extend(model)
    optimizer = DP_SGD(model.parameters(), lr=args.learning_rate, **dpsgd_kwargs[args.experiment])
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
            with backpack(BatchGrad(), BatchL2Grad()):
                loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        print("Time Taken for Epoch: ", duration)
        timings.append(duration)

    if not args.no_save:
        utils.save_runtimes(__file__.split('.')[0], args, timings)
    else:
        print('Not saving!')
    print('Done!')


if __name__ == '__main__':
    parser = utils.get_parser(dpsgd_kwargs.keys())
    args = parser.parse_args()
    main(args)
