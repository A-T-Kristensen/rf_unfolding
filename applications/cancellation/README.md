# Full-Duplex Self-Interference Cancellation

This directory contains the source-code used for the non-linear digital self-interference cancellation for in-band full-duplex radios using neural networks.

The different `run_<>.py` files apply different methods for self-interference cancelletation:

* [run_complex_ffnn.py](run_complex_ffnn.py):  Perform SI cancellation using the complex-valued neural network.
* [run_complex_rnn.py](run_complex_rnn.py):  Perform SI cancellation using the complex-valued RNN.
* [run_ffnn.py](run_ffnn.py):  Perform SI cancellation using the real-valued neural network.
* [run_model_based_nn.py](run_model_based_nn.py):  Perform SI cancellation using the model-based neural network.
* [run_rnn.py](run_rnn.py):  Perform SI cancellation using the recurrent neural network.
* [run_wlmp.py](run_wlmp.py): Perform SI cancellation using the polynomial model (baseline).

The directory [helpers](helpers) contains various helper functions for the experiments.

## How-To

To run, e.g., the model-based NN, you simply run the following command
```
python run_model_based_nn.py --max-power=5 --fit-option=all --n-epochs=50 --batch-size=6 --learning-rate=0.25 --optimizer=ftrl
```
This will generate a model with powers {1,3,5} and learn to perform the full SI cancellation by training the model for 50 epochs, with a batch-size of 6, a learning-rate of 0.25 and using the FTRL optimizer.

If you, e.g., want to train the complex-valued NN, you can do
```
python run_complex_ffnn.py --ffnn-struct=10 --fit-option=nl --n-epochs=50 --batch-size=4 --learning-rate=1e-3 --optimizer=adam
```
This will train a complex-valued neural network with 1 hidden layer of 10 complex-valued neurons, train it to perform cancellation after linear cancellation has first been applied.

If you want to generate the results used for the different papers, simply run the command in the next sections.

### Identification of Non-Linear RF Systems Using Backpropagation

To just generate the model-based NN results, do the following:

```
make hammerstein_ftrl
```

### Advanced Machine Learning Techniques for Self-Interference Cancellation in Full-Duplex Radios

These commands will generate the results for this paper and *Advanced Machine Learning Techniques for Self-Interference Cancellation in Full-Duplex Radios*

```
make run_all
make activation_adam
make learn_all_adam
make wlmp
```

The results will be in the [results](./results) directory.

Not that these results may differ slightly from those in the paper due to some code modifications performed along the development process.

