# Enabling Fast Differentially Private SGD via Static Graph Compilation and Batch-Level Parallelism

The results were obtained on a Titan V GPU on Ubuntu 18.04, CUDA 11.0, and Python 3.8.5. The code will run on CUDA 10.2+ and Python 3.6+. We advise creating a fresh `pip` environment then installing the requirements, as follows:

```
# install jaxlib
PYTHON_VERSION=cp37  # alternatives: cp36, cp37, cp38
CUDA_VERSION=cuda100  # alternatives: cuda100, cuda101, cuda102, cuda110
PLATFORM=manylinux2010_x86_64  # alternatives: manylinux2010_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.55-$PYTHON_VERSION-none-$PLATFORM.whl

pip install -r requirements.txt
```

Before running experiments, ensure the `./results/raw` folder exists in this directory.

To run the ablation study and runtime performance experiments, use the following commands from this directory:

```
python runtime_experiment.py
```

To run the memory limit experiment, use the below commands. You may need to set `--init_bs` to a smaller value if you are using a smaller GPU.

```
python memory_experiment.py --experiments mnist --thresh 128
python memory_experiment.py --experiments lstm --thresh 1
```

The notebooks in the `./results` folder have code to process the emitted pickle files and produce the plots found in the paper.

## XLA Dumps

There are a few XLA dumps present within the supplement. The first is the folder titled `text_xla_dumps`. This is for the synthetic experiment. The first model considered is a matrix-vector product. In pure numpy it can be expressed with:

```
W = np.random.randn(5,5)  # a 5x5 random matrix (from a Multivariate Normal dist.)
x = np.random.randn(5)
b = np.random.randn(5)  # the offset/bias term

@jax.jit
def matvecprod(W, x, b):
    return np.dot(W, x) + b
```

The equivalent TensorFlow variant is expressed as well. These files are: `jax_nograd_linear.txt` and `tf_nograd_linear.txt`. Then the gradients are taken with respect to `W` and `b` of a mean-squared-error loss. These files are `jax_linear.txt` and `tf_linear.txt`. In this file we notice differences in the XLA dump produced by the two frameworks. In this setting JAX is faster by a small margin compared to TensorFlow 2 + XLA.

The dumps for the fully connected network are placed in a file titled `dumps.zip`. According to the TensorFlow documentation, one module is generated for each compiled cluster. These logs are quite verbose, but at the bottom of the final module generated there is a function titled "Entry" which represents the entry point to the program. From there, a reader can trace the called functions and notice immediate differences. An example of such a difference is that JAX has a different number of fused kernels compared to TensorFlow.
