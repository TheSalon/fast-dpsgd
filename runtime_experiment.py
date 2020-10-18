import argparse
import pprint
import subprocess

from utils import pr_green, pr_red


def launch(expt, batch_size, epochs):
    """Runs expt at batch_size for all the scripts"""
    errors = []
    # yapf: disable
    cmds = [
        ('jax', f'CUDA_VISIBLE_DEVICES=0 python jaxdp.py {expt} --no_dpsgd --epochs {epochs} --batch_size {batch_size}'),
        ('tf2', f'CUDA_VISIBLE_DEVICES=0 python tf2dp.py {expt} --no_dpsgd --epochs {epochs} --batch_size {batch_size} --no_xla'),
        ('tf1', f'CUDA_VISIBLE_DEVICES=0 python tf1dp.py {expt} --no_dpsgd --epochs {epochs} --batch_size {batch_size} --no_xla'),
        ('pytorch', f'CUDA_VISIBLE_DEVICES=0 python pytorch.py {expt} --no_dpsgd --epochs {epochs} --batch_size {batch_size}'),
        ('jaxdp', f'CUDA_VISIBLE_DEVICES=0 python jaxdp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size}'),
        ('tf2dp', f'CUDA_VISIBLE_DEVICES=0 python tf2dp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size} --no_xla'),
        ('tf1dp', f'CUDA_VISIBLE_DEVICES=0 python tf1dp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size} --no_xla'),
        ('opacusdp', f'CUDA_VISIBLE_DEVICES=0 python opacusdp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size}'),
        ('backpackdp', f'CUDA_VISIBLE_DEVICES=0 python backpackdp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size}'),
        ('pyvacydp', f'CUDA_VISIBLE_DEVICES=0 python pyvacydp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size}'),
        ('owkindp', f'CUDA_VISIBLE_DEVICES=0 python owkindp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size}'),
        ('tf2xla', f'CUDA_VISIBLE_DEVICES=0 python tf2dp.py {expt} --no_dpsgd --epochs {epochs} --batch_size {batch_size} --xla'),
        ('tf2dpxla', f'CUDA_VISIBLE_DEVICES=0 python tf2dp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size} --xla'),
        ('tf1xla', f'TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=0 python tf1dp.py {expt} --no_dpsgd --epochs {epochs} --batch_size {batch_size} --xla'),
        ('tf1dpxla', f'TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=0 python tf1dp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size} --xla'),
        # For Ablations:
        ('jaxdp_nv', f'CUDA_VISIBLE_DEVICES=0 python jaxdp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size} --no_vmap'),
        # Outside of JIT compilation, the dynamic_unroll's LSTM (using scan) is faster than the static_unroll'd version.
        ('jaxdp_nj', f'CUDA_VISIBLE_DEVICES=0 python jaxdp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size} --no_jit --dynamic_unroll'),
        ('jaxdp_nvj', f'CUDA_VISIBLE_DEVICES=0 python jaxdp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size} --no_vmap --no_jit --dynamic_unroll'),
        ('tf2dp_nv', f'CUDA_VISIBLE_DEVICES=0 python tf2dp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size} --no_xla --no_vmap'),
        ('tf2dp_nvj', f'CUDA_VISIBLE_DEVICES=0 python tf2dp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size} --no_xla --no_vmap --no_jit'),
        ('tf2dpxla_nv', f'CUDA_VISIBLE_DEVICES=0 python tf2dp.py {expt} --dpsgd --epochs {epochs} --batch_size {batch_size} --xla --no_vmap'),
    ]
    # yapf: enable
    for name, cmd in cmds:
        if expt == 'lstm':
            if 'jax' in name:
                # Due to https://github.com/deepmind/dm-haiku/issues/77, we disable
                # omnistaging when running the LSTM in JAX (it will fail to compile).
                cmd = 'JAX_OMNISTAGING=0 ' + cmd
            if name in ('tf1', 'tf2', 'tf1xla', 'tf2dp_nv'):
                # The dynamically unrolled LSTM uses the cudNN LSTM implementation
                # in the non-vectorized_map case, making it faster.
                cmd = cmd + ' --no_unroll'
        pr_green(f'Starting {name}: {cmd}')
        out = subprocess.run([cmd],
                             shell=True,
                             stderr=subprocess.STDOUT,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
        print(out.stdout)
        if out.returncode != 0:
            errors.append(name)
            pr_red(out.stderr)
            print()
            pr_red(f'Done {name}: {cmd} \n')
        else:
            pr_green(f'Done {name}: {cmd} \n')
    pr_green(f'Done {expt} at batch size {batch_size}.')
    return errors


def main(args):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(args)
    failed = {}
    for expt in args.experiments:
        for bs in args.batch_sizes:
            failed[(expt, bs)] = launch(expt, bs, args.epochs)
    pr_red('\nFailed Experiments: \n')
    pp.pprint(failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Returns Max Batch Size before OOM')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--experiments',
                        default=['logreg', 'ffnn', 'mnist', 'embed', 'lstm', 'cifar10'],
                        nargs='+')
    parser.add_argument('--batch_sizes', default=[256, 128, 64, 32, 16], nargs='+', type=int)
    args = parser.parse_args()
    main(args)
