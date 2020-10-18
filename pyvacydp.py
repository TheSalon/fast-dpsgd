'''
Pyvacy implementations
'''

import time

import torch
import torch.nn.functional as F
from pyvacy import analysis, optim
from torch import nn

import data
import utils
from pytorch import get_data, model_dict


def main(args):
    print(args)
    assert args.dpsgd
    torch.backends.cudnn.benchmark = True

    train_data, train_labels = get_data(args)
    num_complete_batches, leftover = divmod(len(train_data), args.batch_size)
    num_batches = num_complete_batches + bool(leftover)

    model = model_dict[args.experiment](vocab_size=args.max_features).cuda()
    loss_function = nn.CrossEntropyLoss() if args.experiment != 'logreg' else nn.BCELoss()

    opt = optim.DPSGD(params=model.parameters(),
                      l2_norm_clip=args.l2_norm_clip,
                      noise_multiplier=args.noise_multiplier,
                      minibatch_size=args.batch_size,
                      microbatch_size=1,
                      lr=args.learning_rate)

    timings = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        dataloader = data.dataloader(train_data, train_labels, args.batch_size)
        for batch_idx, (x_mb, y_mb) in enumerate(dataloader):
            x_mb, y_mb = x_mb.cuda(non_blocking=True), y_mb.cuda(non_blocking=True)
            for x, y in zip(x_mb, y_mb):
                opt.zero_microbatch_grad()
                out = model(x[None])
                curr_loss = loss_function(out, y[None])
                curr_loss.backward()
                opt.microbatch_step()
            opt.step()
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
