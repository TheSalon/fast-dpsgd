import argparse
import pickle
import subprocess

from utils import pr_green, pr_red

# yapf: disable
CMDS = dict((
    ('jax',        'python jaxdp.py {} --no_dpsgd --no_save --dummy_data'),
    ('tf2',        'python tf2dp.py {} --no_dpsgd --no_xla --no_save --dummy_data'),
    ('tf1',        'python tf1dp.py {} --no_dpsgd --no_xla --no_save --dummy_data'),
    ('pytorch',    'python pytorch.py {} --no_dpsgd --no_save --dummy_data'),
    ('jaxdp',      'python jaxdp.py {} --dpsgd --no_save --dummy_data'),
    ('tf2dp',      'python tf2dp.py {} --dpsgd --no_xla --no_save --dummy_data'),
    ('tf1dp',      'python tf1dp.py {} --dpsgd --no_xla --no_save --dummy_data'),
    ('opacusdp',   'python opacusdp.py {} --dpsgd --no_save --dummy_data'),
    ('backpackdp', 'python backpackdp.py {} --dpsgd --no_save --dummy_data'),
    ('owkindp',    'python owkindp.py {} --dpsgd --no_save --dummy_data'),
    ('tf2xla',     'python tf2dp.py {} --no_dpsgd --xla --no_save --dummy_data'),
    ('tf2dpxla',   'python tf2dp.py {} --dpsgd --xla --no_save --dummy_data'),
    ('tf1xla',     'TF_XLA_FLAGS=--tf_xla_auto_jit=2 python tf1dp.py {} --no_dpsgd --xla --no_save --dummy_data'),
    ('tf1dpxla',   'TF_XLA_FLAGS=--tf_xla_auto_jit=2 python tf1dp.py {} --dpsgd --xla --no_save --dummy_data'),

    # PyVacy processes examples individually irrespective of batch size, so it won't OOM, so we don't test it.
    # ('pyvacydp',   'python pyvacydp.py {} --dpsgd --no_save --dummy_data'),
))
# yapf: enable


def oom_fn(bs, cmd, print_error=False):
    """Runs script at batch size bs and checks if the script OOMs"""
    proc = subprocess.run(
        [cmd + f' --batch_size {bs}'],
        # check=True,
        shell=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        universal_newlines=True)

    lower = proc.stdout.lower()
    # last condiiton is because of a (hard to reproduce) Pytorch bug. When the batch size
    # is slightly too big, we'll get a CuDNN error instead of an OOM error.
    oom = ('out of memory' in lower or 'oom' in lower or 'resourceexhausted' in lower
           or 'cudnn error' in lower)
    if oom and print_error:
        pr_red(proc.stdout)
        pr_red(proc.stderr)
    if not oom and proc.returncode != 0:
        pr_red(proc.stdout)
        pr_red(proc.stderr)
        raise ValueError('Not OOM but returncode != 0')

    s = '' if oom else 'not'
    print(f'Batch Size {bs} {s} OOM.')
    return oom


def binary_search(low, high, cmd, args):
    if high - low > args.thresh:
        mid = int((high + low) // 2)
        oom = oom_fn(mid, cmd)
        if oom:
            return binary_search(low, mid, cmd, args)
        else:
            return binary_search(mid, high, cmd, args)
    else:
        return low


def get_max_batchsize(run, expt, args):
    bs = args.init_bs
    oom = False

    cmd = f'CUDA_VISIBLE_DEVICES={args.device} {CMDS[run].format(expt)} --epochs {args.epochs}'
    if expt == 'lstm':
        if 'jax' in run:
            cmd = 'JAX_OMNISTAGING=0 ' + cmd
        if run in ('tf1', 'tf2', 'tf1xla'):
            cmd = cmd + ' --no_unroll'
    pr_green(cmd)
    out = subprocess.run([cmd + f' --batch_size {bs}'],
                         shell=True,
                         stderr=subprocess.STDOUT,
                         stdout=subprocess.PIPE,
                         universal_newlines=True).stdout
    print(out)
    if 'Error' in out:
        return (-1, -1)

    # Get a reasonable range for the batch size
    while not oom:
        bs *= 2
        oom = oom_fn(bs, cmd, print_error=True)

    max_bs = binary_search(bs / 2, bs, cmd, args)
    pr_green(f'Max Batch Size: {max_bs}')
    return (max_bs, max_bs + args.thresh)


def main(args):
    print(args)
    name = '_' + args.name if args.name else ''

    save_list = []
    for run in args.runs:
        for expt in args.experiments:
            save_list.append((run, expt, *get_max_batchsize(run, expt, args)))
            with open(f'results/raw/memory_expt{name}.pkl', 'wb') as handle:
                pickle.dump(save_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'results/raw/memory_expt{name}.pkl', 'wb') as handle:
        pickle.dump(save_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done! Saved to results/raw/memory_expt{name}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Returns Max Batch Size before OOM')
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--init_bs', default=64, type=int)
    parser.add_argument('--thresh', default=8, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--experiments',
                        default=['logreg', 'ffnn', 'mnist', 'embed', 'lstm', 'cifar10'],
                        nargs='+')
    parser.add_argument('--runs', default=CMDS.keys(), nargs='+')
    args = parser.parse_args()
    main(args)
