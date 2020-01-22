# Full-Duplex Self-Interference Cancellation

This directory contains the source-code used for the non-linear digital self-interference cancellation for in-bank full-duplex radios using neural networks.

The different `run_<>.py` files apply different methods for self-interference cancelletation.
The directory [helpers](helpers) contains various helper functions for the experiments.

## How-To

If you want to generate the results used for the different papers, do

### Advanced Machine Learning Techniques for Self-Interference Cancellation in Full-Duplex Radios

These commands will generate the results for this paper and *Identification of Non-Linear RF Systems Using Backpropagation*

```
make run_all
make activation_adam
make learn_all_adam
make wlmp
```
The results will be in the [results](./results) directory.

Not that these results may differ slightly from those in the paper due to some
code modifications performed along the development process.

### Identification of Non-Linear RF Systems Using Backpropagation

To just generate the model-based NN results, do the following:

```
make hammerstein_ftrl
```

## Cancellers

### 1: Polynomial Cancellation

* [run_poly.py](run_poly.py): This script loads the measured testbed data and performs non-linear cancellation using the baseline polynomial model.

### 2: Real-Valued Feed-Forward Neural Network

* [run_ffnn.py](run_ffnn.py): This script loads the measured testbed data and performs non-linear cancellation using the neural network based method proposed in [1].

### 3: Complex-Valued Feed-Forward Neural Network

* [run_complex_ffnn.py](run_complex_ffnn.py): This script applies a complex-valued neural network for the cancellation.

### 4: Recurrent Neural-Network

* [run_rnn.py](run_rnn.py): This script applies a recurrent neural network for the cancellation.

### 5: Complex-Valued Recurrent Neural-Network

* [run_complex_rnn.py](run_complex_rnn.py): This script applies a complex-valued recurrent neural network for the cancellation.

### 6: Model-Based Neural Network

* [run_hammerstein.py](run_hammerstein.py): This script applies a complex-valued model-based neural network to perform the cancellation.

# References

[1] A. Balatsoukas-Stimming, "Non-linear digital self-interference cancellation for in-band full-duplex radios using neural networks," in IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), Jun. 2018