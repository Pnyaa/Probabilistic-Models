# Energy-based models and Restricted Boltzmann Machine

The [EBM_RBM notebook](./EBM_RBM.ipynb) contains both the theoretical explanations of Energy-based models and Restricted Boltzmann Machines, and several Python implementations of such models for different set of hyperparameters. <br>
This work was developed on Google Colab (hence the weird indentations) and served as an introduction to the Pytorch framework. The notebook comes with a number of folders, each containing parameters of models trained under different conditions :
 * **batch_sizes** contains the parameters of models trained with different batch sizes, with the other hyperparameters being held constant (batch sizes varying between 1 and 5000)
 * **hidden_units** contains the parameters of models trained with different numbers of hidden units, with the other hyperparameters being held constant (number of hidden units varying between 10 and 1000)
 * **iter_samples** contains the parameters of models trained with different number of iterations in the Gibbs sampler, with the other hyperparameters being held constant (number of iterations varying between 1 and 150)
 * **learning_rates** contains the parameters of models trained with different learning rates, with the other hyperparameters being held constant (learning rates varying between 0.00001 and 300)


The functions to create functional RBM from these sets of parameters are provided in the notebook. <br>
The folder **best_model** contains the parameters of the model that gave best performances according to the metrics defined<br>
The **handmade** folder contains the images of manually drawn numbers used for the tests at the end of the notebook.
