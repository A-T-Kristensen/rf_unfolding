#--------------------------------------------------------------------------#
#
# AUTHOR:   Andreas Toftegaard Kristensen
#
# PURPOSE:  The Makefile is used for executing all the different experiments.
#
# EXAMPLE:
#       Run all main experiments (default)
#       > make test_all
#
#--------------------------------------------------------------------------#

SHELL := /bin/bash

#----------------------------- VARIABLES ----------------------------------#

.DEFAULT_GOAL := run_all

# Remember, the upper limit is exclusive
# Setting the maximum learning-rate too high leads to finding more slightly unstable hyperparameters

# Simplify these, the only difference is the chosen optimizer, in the end, we did not need special values for the diff optimizer
HTNN_OPT_ADADELTA = --min-batch-size=4 --max-batch-size=16 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=adadelta
NN_OPT_ADADELTA = --min-batch-size=4 --max-batch-size=64 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=adadelta

HTNN_OPT_ADAGRAD = --min-batch-size=4 --max-batch-size=16 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=adagrad
NN_OPT_ADAGRAD = --min-batch-size=4 --max-batch-size=64 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=adagrad

HTNN_OPT_ADAM = --min-batch-size=4 --max-batch-size=16 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=adam
NN_OPT_ADAM = --min-batch-size=4 --max-batch-size=64 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=adam
RNN_OPT_ADAM = --min-batch-size=16 --max-batch-size=256 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --rnn-unroll=1 --optimizer=adam

HTNN_OPT_ADAMAX = --min-batch-size=4 --max-batch-size=16 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=adamax
NN_OPT_ADAMAX = --min-batch-size=4 --max-batch-size=64 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=adamax

HTNN_OPT_FTRL = --min-batch-size=4 --max-batch-size=16 --min-learning-rate=1e-3 --max-learning-rate=5e-1 --optimizer=ftrl
NN_OPT_FTRL = --min-batch-size=4 --max-batch-size=64 --min-learning-rate=1e-3 --max-learning-rate=5e-1 --optimizer=ftrl
RNN_OPT_FTRL = --min-batch-size=16 --max-batch-size=256 --min-learning-rate=1e-3 --max-learning-rate=5e-1 --rnn-unroll=1 --optimizer=ftrl

HTNN_OPT_NADAM = --min-batch-size=4 --max-batch-size=16 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=nadam
NN_OPT_NADAM = --min-batch-size=4 --max-batch-size=64 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=nadam

HTNN_OPT_RMSPROP = --min-batch-size=4 --max-batch-size=32 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=rmsprop
NN_OPT_RMSPROP = --min-batch-size=4 --max-batch-size=64 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=rmsprop

HTNN_OPT_SGD = --min-batch-size=4 --max-batch-size=16 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=sgd
NN_OPT_SGD = --min-batch-size=4 --max-batch-size=64 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --optimizer=sgd
RNN_OPT_SGD = --min-batch-size=16 --max-batch-size=256 --min-learning-rate=1e-4 --max-learning-rate=5e-1 --rnn-unroll=1 --optimizer=sgd

HTNN_OPT_SGD_SCHEDULE = --min-batch-size=4 --max-batch-size=16 --min-learning-rate=1e-3 --max-learning-rate=5e-1 --optimizer=sgd --lr-schedule=1
NN_OPT_SGD_SCHEDULE = --min-batch-size=4 --max-batch-size=64 --min-learning-rate=1e-3 --max-learning-rate=5e-1 --optimizer=sgd --lr-schedule=1
RNN_OPT_SGD_SCHEDULE = --min-batch-size=16 --max-batch-size=256 --min-learning-rate=1e-3 --max-learning-rate=5e-1 --rnn-unroll=1 --optimizer=sgd --lr-schedule=1

#----------------------------- TEST ALL ----------------------------------#

# Generic options for search
SEARCH_OPT_TEST = --search=1 --n-search-points=2
MP_OPT_TEST= --mp=1 --n-jobs=-1
CV_OPT_TEST= --cv=1 --cv-folds=2 --n-cv-seeds=2 --n-cv-epochs=2
TRAIN_OPT_TEST = --n-epochs=2 --n-seeds=2 --search-training-size=0
COMMON_OPT_TEST = $(SEARCH_OPT_TEST) $(MP_OPT_TEST) $(CV_OPT_TEST) $(TRAIN_OPT_TEST) --save=0 --exp-name=test_scripts

test_all:
	python run_hammerstein.py $(COMMON_OPT_TEST) $(HTNN_OPT_FTRL) --htnn-struct=T1-C
	python run_hammerstein.py $(COMMON_OPT_TEST) $(HTNN_OPT_FTRL) --htnn-struct=T2-C
	python run_hammerstein.py $(COMMON_OPT_TEST) $(HTNN_OPT_FTRL) --htnn-struct=E-C
	python run_hammerstein.py $(COMMON_OPT_TEST) $(HTNN_OPT_FTRL) --htnn-struct=C-C
	python run_hammerstein.py $(COMMON_OPT_TEST) $(HTNN_OPT_FTRL) --htnn-struct=R-C
	python run_ffnn.py $(COMMON_OPT_TEST) $(NN_OPT_ADAM) --ffnn-struct=1
	python run_ffnn.py $(COMMON_OPT_TEST) $(NN_OPT_ADAM) --ffnn-struct=1-1-1
	python run_complex_ffnn.py $(COMMON_OPT_TEST) $(NN_OPT_ADAM) --ffnn-struct=1
	python run_complex_ffnn.py $(COMMON_OPT_TEST) $(NN_OPT_ADAM) --ffnn-struct=1-1-1
	python run_rnn.py $(COMMON_OPT_TEST) $(RNN_OPT_ADAM) --ffnn-struct=1 --rnn-cell=simple
	python run_rnn.py $(COMMON_OPT_TEST) $(RNN_OPT_ADAM) --ffnn-struct=1-1-1 --rnn-cell=simple
	python run_complex_rnn.py $(COMMON_OPT_TEST) $(RNN_OPT_ADAM) --ffnn-struct=1 --rnn-cell=simple
	python run_complex_rnn.py $(COMMON_OPT_TEST) $(RNN_OPT_ADAM) --ffnn-struct=1-1-1 --rnn-cell=simple

#--------------------------- MAIN EXPERIMENTS --------------------------------#

# Generic options
SEARCH_OPT = --search=1 --n-search-points=20
MP_OPT = --mp=1 --n-jobs=-1
CV_OPT = --cv=1 --cv-folds=5 --n-cv-seeds=5 --n-cv-epochs=20
TRAIN_OPT = --n-epochs=50 --n-seeds=20 --search-training-size=0
COMMON_OPT = $(SEARCH_OPT) $(MP_OPT) $(CV_OPT) $(TRAIN_OPT) --save=1

# FTRL Optimizer
complex_ffnn_ftrl:
	python run_complex_ffnn.py $(COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=1 --search-width=1 --min-width=2 --max-width=10 --step-width=1
	python run_complex_ffnn.py $(COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=1-1-1 --search-width=1 --min-width=2 --max-width=10 --step-width=1

ffnn_ftrl:
	python run_ffnn.py $(COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=1 --search-width=1 --min-width=2 --max-width=20 --step-width=2
	python run_ffnn.py $(COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=1-1-1 --search-width=1 --min-width=2 --max-width=20 --step-width=2

hammerstein_ftrl:
	python run_hammerstein.py $(COMMON_OPT) $(HTNN_OPT_FTRL) --htnn-struct=C-C --search-width=1 --min-width=3 --max-width=9 --step-width=2
	python run_hammerstein.py $(COMMON_OPT) $(HTNN_OPT_FTRL) --htnn-struct=R-C --search-width=1 --min-width=3 --max-width=9 --step-width=2
	python run_hammerstein.py $(COMMON_OPT) $(HTNN_OPT_FTRL) --htnn-struct=B-C --search-width=1 --min-width=3 --max-width=9 --step-width=2

complex_rnn_ftrl:
	python run_complex_rnn.py $(COMMON_OPT) $(RNN_OPT_FTRL) --ffnn-struct=1 --rnn-cell=simple --search-width=1 --min-width=2 --max-width=20 --step-width=2
	python run_complex_rnn.py $(COMMON_OPT) $(RNN_OPT_FTRL) --ffnn-struct=1-1-1 --rnn-cell=simple --search-width=1 --min-width=2 --max-width=20 --step-width=2

rnn_ftrl:
	python run_rnn.py $(COMMON_OPT) $(RNN_OPT_FTRL) --ffnn-struct=1 --rnn-cell=simple --search-width=1 --min-width=2 --max-width=20 --step-width=2
	python run_rnn.py $(COMMON_OPT) $(RNN_OPT_FTRL) --ffnn-struct=1-1-1 --rnn-cell=simple --search-width=1 --min-width=2 --max-width=20 --step-width=2

# ADAM Optimizer
complex_ffnn_adam:
	python run_complex_ffnn.py $(COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=1 --search-width=1 --min-width=2 --max-width=10 --step-width=1
	python run_complex_ffnn.py $(COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=1-1-1 --search-width=1 --min-width=2 --max-width=10 --step-width=1

ffnn_adam:
	python run_ffnn.py $(COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=1 --search-width=1 --min-width=2 --max-width=20 --step-width=2
	python run_ffnn.py $(COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=1-1-1 --search-width=1 --min-width=2 --max-width=20 --step-width=2

hammerstein_adam:
	python run_hammerstein.py $(COMMON_OPT) $(HTNN_OPT_ADAM) --htnn-struct=C-C --search-width=1 --min-width=3 --max-width=9 --step-width=2
	python run_hammerstein.py $(COMMON_OPT) $(HTNN_OPT_ADAM) --htnn-struct=R-C --search-width=1 --min-width=3 --max-width=9 --step-width=2

complex_rnn_adam:
	python run_complex_rnn.py $(COMMON_OPT) $(RNN_OPT_ADAM) --ffnn-struct=1 --rnn-cell=simple --search-width=1 --min-width=2 --max-width=20 --step-width=2
	python run_complex_rnn.py $(COMMON_OPT) $(RNN_OPT_ADAM) --ffnn-struct=1-1-1 --rnn-cell=simple --search-width=1 --min-width=2 --max-width=20 --step-width=2

rnn_adam:
	python run_rnn.py $(COMMON_OPT) $(RNN_OPT_ADAM) --ffnn-struct=1 --rnn-cell=simple --search-width=1 --min-width=2 --max-width=20 --step-width=2
	python run_rnn.py $(COMMON_OPT) $(RNN_OPT_ADAM) --ffnn-struct=1-1-1 --rnn-cell=simple --search-width=1 --min-width=2 --max-width=20 --step-width=2

# Polynomial

wlmp:
	python run_wlmp.py --save=1 --search-width=1 --min-width=1 --max-width=9 --step-width=2

# Execute All
run_all_ftrl: hammerstein_ftrl
run_all_adam: complex_ffnn_adam ffnn_adam hammerstein_adam rnn_adam
run_all: run_all_ftrl run_all_adam


#--------------------------- HAMMERSTEIN TYPE --------------------------------#

# Experiments for the type of IQ-mixer layer
COMMON_OPT_HAMMERSTEIN_TYPE = $(SEARCH_OPT) $(MP_OPT) $(CV_OPT) $(TRAIN_OPT) --save=0 --exp-name=htnn_type --search-width=1 --min-width=3 --max-width=9 --step-width=2

hammerstein_type:
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_TYPE) $(HTNN_OPT_FTRL) --htnn-struct=T1-C --gradient-clipping-param1=0.0 --gradient-clipping-param2=0.0 --gradient-clipping=0.0 --initializer=hammerstein_rayleigh
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_TYPE) $(HTNN_OPT_FTRL) --htnn-struct=T2-C --gradient-clipping-param1=0.0 --gradient-clipping-param2=0.0 --gradient-clipping=0.0 --initializer=hammerstein_rayleigh
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_TYPE) $(HTNN_OPT_FTRL) --htnn-struct=E-C --gradient-clipping-param1=0.0 --gradient-clipping-param2=0.0 --gradient-clipping=0.0 --initializer=hammerstein_rayleigh
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_TYPE) $(HTNN_OPT_FTRL) --htnn-struct=C-C --gradient-clipping-param1=0.0 --gradient-clipping-param2=0.0 --gradient-clipping=0.0 --initializer=hammerstein_rayleigh
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_TYPE) $(HTNN_OPT_FTRL) --htnn-struct=R-C --gradient-clipping-param1=0.0 --gradient-clipping-param2=0.0 --gradient-clipping=0.0 --initializer=hammerstein_rayleigh
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_TYPE) $(HTNN_OPT_FTRL) --htnn-struct=B-C --gradient-clipping=0.0 --initializer=hammerstein_rayleigh


# Experiments for initialization of Hammerstein, assuming we already have
COMMON_OPT_HAMMERSTEIN_INIT = $(SEARCH_OPT) $(MP_OPT) $(CV_OPT) $(TRAIN_OPT) --save=0 --exp-name=htnn_init --search-width=1 --min-width=3 --max-width=9 --step-width=2

hammerstein_init:
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_INIT) $(HTNN_OPT_FTRL) --htnn-struct=C-C --gradient-clipping-param1=0.0 --gradient-clipping-param2=0.0 --gradient-clipping=0.0 --initializer=hammerstein_rayleigh
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_INIT) $(HTNN_OPT_FTRL) --htnn-struct=R-C --gradient-clipping-param1=0.0 --gradient-clipping-param2=0.0 --gradient-clipping=0.0 --initializer=hammerstein_rayleigh
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_INIT) $(HTNN_OPT_FTRL) --htnn-struct=C-C --gradient-clipping-param1=0.0 --gradient-clipping-param2=0.0 --gradient-clipping=0.0 --initializer=normal
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_INIT) $(HTNN_OPT_FTRL) --htnn-struct=R-C --gradient-clipping-param1=0.0 --gradient-clipping-param2=0.0 --gradient-clipping=0.0 --initializer=normal
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_INIT) $(HTNN_OPT_FTRL) --htnn-struct=C-C --gradient-clipping-param1=1.0 --gradient-clipping-param2=0.001 --gradient-clipping=0.1 --initializer=hammerstein_rayleigh
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_INIT) $(HTNN_OPT_FTRL) --htnn-struct=R-C --gradient-clipping-param1=1.0 --gradient-clipping-param2=0.001 --gradient-clipping=0.1 --initializer=hammerstein_rayleigh
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_INIT) $(HTNN_OPT_FTRL) --htnn-struct=C-C --gradient-clipping-param1=1.0 --gradient-clipping-param2=0.001 --gradient-clipping=0.1 --initializer=normal
	python run_hammerstein.py $(COMMON_OPT_HAMMERSTEIN_INIT) $(HTNN_OPT_FTRL) --htnn-struct=R-C --gradient-clipping-param1=1.0 --gradient-clipping-param2=0.001 --gradient-clipping=0.1 --initializer=normal

#----------------------------- ACTIVCATIONS ----------------------------------#

# Acitvation function experiments
ACT_COMMON_OPT = $(SEARCH_OPT) $(MP_OPT) $(CV_OPT) $(TRAIN_OPT) --save=1

activation_adam:
	python run_complex_ffnn.py $(ACT_COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=10 --activation=relu --exp-name=relu
	python run_complex_ffnn.py $(ACT_COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=10 --activation=amp_phase --exp-name=amp_phase
	python run_complex_ffnn.py $(ACT_COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=10 --activation=cardiodid --exp-name=cardiodid
	python run_complex_ffnn.py $(ACT_COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=10 --activation=mrelu --exp-name=mrelu
	python run_complex_ffnn.py $(ACT_COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=10 --activation=zrelu --exp-name=zrelu

activation_ftrl:
	python run_complex_ffnn.py $(ACT_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=10 --activation=relu --exp-name=relu
	python run_complex_ffnn.py $(ACT_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=10 --activation=amp_phase --exp-name=amp_phase
	python run_complex_ffnn.py $(ACT_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=10 --activation=cardiodid --exp-name=cardiodid
	python run_complex_ffnn.py $(ACT_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=10 --activation=mrelu --exp-name=mrelu
	python run_complex_ffnn.py $(ACT_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=10 --activation=zrelu --exp-name=zrelu

#----------------------------- LEARN ALL ----------------------------------#

# Learn all experiments
ALL_COMMON_OPT = $(SEARCH_OPT) $(MP_OPT) $(CV_OPT) $(TRAIN_OPT) --save=0 --exp-name=learn_all --fit-option=all

learn_all_adam:
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=1 --search-width=1 --min-width=5 --max-width=10 --step-width=1
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=1-1-1 --search-width=1 --min-width=5 --max-width=10 --step-width=1
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=1 --search-width=1 --min-width=10 --max-width=20 --step-width=2
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAM) --ffnn-struct=1-1-1 --search-width=1 --min-width=10 --max-width=20 --step-width=2

learn_all_ftrl:
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=1 --search-width=1 --min-width=5 --max-width=10 --step-width=1
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=1-1-1 --search-width=1 --min-width=5 --max-width=10 --step-width=1
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=1 --search-width=1 --min-width=10 --max-width=20 --step-width=2
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=1-1-1 --search-width=1 --min-width=10 --max-width=20 --step-width=2

#----------------------------- OPTIMIZERS ----------------------------------#

# Generate results for all the different optimizers
OPTIMIZER_COMMON_OPT = $(SEARCH_OPT) $(MP_OPT) $(CV_OPT) $(TRAIN_OPT) --save=1

complex_ffnn_optimizer:
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADADELTA) --ffnn-struct=10 --fit-option=all
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAGRAD) --ffnn-struct=10 --fit-option=all
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAMAX) --ffnn-struct=10 --fit-option=all
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_NADAM) --ffnn-struct=10 --fit-option=all
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_RMSPROP) --ffnn-struct=10 --fit-option=all
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_SGD) --ffnn-struct=10 --fit-option=all
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADADELTA) --ffnn-struct=10 --fit-option=nl
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAGRAD) --ffnn-struct=10 --fit-option=nl
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAMAX) --ffnn-struct=10 --fit-option=nl
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_NADAM) --ffnn-struct=10 --fit-option=nl
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_RMSPROP) --ffnn-struct=10 --fit-option=nl
	python run_complex_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_SGD) --ffnn-struct=10 --fit-option=nl

ffnn_optimizer:
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADADELTA) --ffnn-struct=20 --fit-option=all
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAGRAD) --ffnn-struct=20 --fit-option=all
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAMAX) --ffnn-struct=20 --fit-option=all
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=20 --fit-option=all
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_NADAM) --ffnn-struct=20 --fit-option=all
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_RMSPROP) --ffnn-struct=20 --fit-option=all
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_SGD) --ffnn-struct=20 --fit-option=all
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADADELTA) --ffnn-struct=20 --fit-option=nl
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAGRAD) --ffnn-struct=20 --fit-option=nl
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_ADAMAX) --ffnn-struct=20 --fit-option=nl
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_FTRL) --ffnn-struct=20 --fit-option=nl
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_NADAM) --ffnn-struct=20 --fit-option=nl
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_RMSPROP) --ffnn-struct=20 --fit-option=nl
	python run_ffnn.py $(ALL_COMMON_OPT) $(NN_OPT_SGD) --ffnn-struct=20 --fit-option=nl

hammerstein_optimizer:
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_ADADELTA) --htnn-struct=C-C --fit-option=all
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_ADAGRAD) --htnn-struct=C-C --fit-option=all
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_ADAMAX) --htnn-struct=C-C --fit-option=all
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_FTRL) --htnn-struct=C-C --fit-option=all
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_NADAM) --htnn-struct=C-C --fit-option=all
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_RMSPROP) --htnn-struct=C-C --fit-option=all
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_SGD) --htnn-struct=C-C --fit-option=all
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_ADADELTA) --htnn-struct=C-C --fit-option=nl
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_ADAGRAD) --htnn-struct=C-C --fit-option=nl
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_ADAMAX) --htnn-struct=C-C --fit-option=nl
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_FTRL) --htnn-struct=C-C --fit-option=nl
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_NADAM) --htnn-struct=C-C --fit-option=nl
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_RMSPROP) --htnn-struct=C-C --fit-option=nl
	python run_hammerstein.py $(ALL_COMMON_OPT) $(HTNN_OPT_SGD) --htnn-struct=C-C --fit-option=nl

#----------------------------- VARIABLES ----------------------------------#

clean:
	find . -wholename '.*.png' -delete
