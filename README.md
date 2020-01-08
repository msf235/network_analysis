# network_analysis
Tools for the analysis of networks, including trained networks, in Python 3.7. Training is done by pytorch.

There are three main packages

These tools are not currently prepared for use by a wider audience (in particular, documentation is sparse and inconsistent). Future updates may change this.

There are four main modules:
  (1) models.py contains PyTorch torch.nn.Module objects. These are network models that can be used for training.
  (2) model_trainer.py contains a function train_model that takes in a model, optimizer, loss function, etc. and trains a model.
  (3) model_output_manager.py contains tools for recording the runs that have taken place. Any time a network is created and trained, the parameters used for this can be handed to model_output_manager in order to create a new row in a table that records every run. If you go to train a model with the same parameters as before, this utility can automatically load the previous trained model instead. ToDo: Show an example of how this works.
  (4) model_loader_utils.py contains utility functions for loading models over epochs. It can return the hidden unit activations, the weights, or the models themselves over epochs.
