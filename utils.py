import argparse
import pickle


def get_parser(experiments):
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=experiments)
    parser.add_argument('--dpsgd', dest='dpsgd', action='store_true')
    parser.add_argument('--no_dpsgd', dest='dpsgd', action='store_false')
    parser.add_argument('--learning_rate', default=0.15, type=float)
    parser.add_argument('--noise_multiplier', default=1.1, type=float)
    parser.add_argument('--l2_norm_clip', default=1.0, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--microbatches', default=None)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--dummy_data', default=False, action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--no_save', dest='no_save', action='store_true')

    # imdb specific
    parser.add_argument('--max_features', default=10_000, type=int)
    parser.add_argument('--max_len', default=256, type=int)

    return parser


def save_runtimes(filename, args, timings, append_to_name=''):
    d = {'args': args, 'timings': timings}
    pickle_name = f'{filename}_{args.experiment}_bs_{args.batch_size}_priv_{args.dpsgd}'
    if hasattr(args, 'use_xla'):
        pickle_name += f'_xla_{args.use_xla}'
    pickle_name += append_to_name
    full_path = './results/raw/' + pickle_name + '.pkl'
    print('Saving to: ', full_path)

    with open(full_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pr_red(text):
    print("\033[91m{}\033[00m".format(text))


def pr_green(text):
    print("\033[92m{}\033[00m".format(text))
