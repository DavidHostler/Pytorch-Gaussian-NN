# Pytorch-Gaussian-NN
The purpose of this little bit of work is to train a simple neural network in Pytorch to generate an output of datapoints
modelled after a Gaussian/Normal distribution. 
Here the atom_dos becomes x, and the Gaussian distribution is implemented as a Pytorch Variable.
An input of a one-dimensional array is fed into the NN with the expectation of 500 prediction points output,i.e.
the NN closely predicts the behaviour of an actual Gaussian function.

Different activation functions ought to be tried out. Results looked good at first with ReLU(x) but Softmax or Sigmoid may be 
better for the purposes of normalization. 
Initially an SGD optimizer was initially employed as well, although likewise I will experiment with Adam since that is quite popular as well.


Instead of simply choosing randomized values,a weighted noise was introduced, although again this method should be treated with a grain of salt. 
Additionally some variables such as mean and standard deviation are hardcoded, and can be tuned to get different results.
Two instances of training the network have been shown, one for no noise and one case for a significant randomised noise,
with loss and mae printed and also graphed.


