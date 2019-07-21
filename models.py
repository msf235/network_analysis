import typing
from typing import Union, List, Callable
import torch
from torch import nn
import math

class FeedForward(nn.Module):
    """
    Class for feedforward neural network model. Takes a list of pytorch tensors and ties these together into a
    trainable neural network. See classes that inheret from this class for more user-friendly options.
    """

    def __init__(self, layers: List[torch.Tensor], biases: List[torch.Tensor], nonlinearities: List[Callable]):
        """
        
        Parameters
        ----------
        layers : 
        biases : 
        nonlinearities : 
        """
        super().__init__()
        self.layers = nn.ParameterList([nn.Parameter(layer, requires_grad=True) for layer in layers])
        self.biases = nn.ParameterList([nn.Parameter(bias, requires_grad=True) for bias in biases])
        self.nonlinearities = nonlinearities

    def forward(self, inputs):
        hid = inputs
        for layer, nonlinearity, bias in zip(self.layers, self.nonlinearities, self.biases):
            hid = nonlinearity(hid @ layer + bias)
        return hid

    def get_pre_activations(self, inputs):
        hid = inputs
        pre_activations = []
        for layer, nonlinearity, bias in zip(self.layers, self.nonlinearities, self.biases):
            pre_activation = hid @ layer + bias
            hid = nonlinearity(pre_activation)
            pre_activations.append(pre_activation.detach())
        return pre_activations

    def get_post_activation(self, inputs):
        hid = inputs
        post_activations = []
        for layer, nonlinearity, bias in zip(self.layers, self.nonlinearities, self.biases):
            hid = nonlinearity(hid @ layer + bias)
            post_activations.append(hid.detach())
        return post_activations

    def get_activations(self, inputs):
        hid = inputs
        activations = []
        for layer, nonlinearity, bias in zip(self.layers, self.nonlinearities, self.biases):
            pre_activation = hid @ layer + bias
            hid = nonlinearity(pre_activation)
            activations.append(pre_activation.detach())
            activations.append(hid.detach())
        return activations

class DenseRandomFF(FeedForward):
    """
    Feedforward net. Weights are initialized according to two
    factors: how close to an "identity" matrix the weights are, and a gain factor. Biases are initialized to zero.
    """

    def __init__(self, input_dim: int,
                 hidden_dims: Union[int, List[int]],
                 output_dim: int,
                 num_layers: int,
                 pert_factor: float = 1.,
                 gain_factor: float = 0.4,
                 nonlinearity: typing.Union[str, typing.Callable] = 'relu',
                 normalize: bool = True):

        if isinstance(nonlinearity, typing.Callable):
            self.nonlinearity = nonlinearity
        elif isinstance(nonlinearity, str):
            if nonlinearity == 'tanh' or nonlinearity == 'Tanh':
                self.nonlinearity = torch.tanh
            elif nonlinearity == 'relu' or nonlinearity == 'ReLU':
                def relu(x):
                    return torch.clamp(x, min=9)

                self.nonlinearity = relu
            else:
                raise AttributeError("nonlinearity not recognized.")
        else:
            raise AttributeError("nonlinearity not recognized.")

        layers = []
        nonlinearities = []
        biases = []

        if not hasattr(hidden_dims, '__len__'):
            N = hidden_dims
            hidden_dims = [N for x in range(num_layers)]
        else:
            if len(hidden_dims) != num_layers:
                raise ValueError("Length of hidden_dims does not match num_layers")

        # input weights
        input_w_id = torch.eye(input_dim, hidden_dims[0])
        if normalize:
            input_w_random = gain_factor * torch.randn(input_dim, hidden_dims[0]) / math.sqrt(input_dim)
        else:
            input_w_random = gain_factor * torch.randn(input_dim, hidden_dims[0])

        input_w = (1 - pert_factor) * input_w_id + pert_factor * input_w_random
        layers.append(input_w)
        nonlinearities.append(self.nonlinearity)
        biases.append(torch.zeros(hidden_dims[0]))

        # hidden layer weights
        for i0 in range(num_layers - 1):
            hidden_w_id = torch.eye(hidden_dims[0], hidden_dims[1])
            if normalize:
                hidden_w_random = gain_factor * torch.randn(hidden_dims[i0], hidden_dims[i0 + 1]) / math.sqrt(
                    hidden_dims[i0])
            else:
                hidden_w_random = gain_factor * torch.randn(hidden_dims[i0], hidden_dims[i0 + 1])

            hidden_w = (1 - pert_factor) * hidden_w_id + pert_factor * hidden_w_random
            layers.append(hidden_w)
            nonlinearities.append(self.nonlinearity)
            biases.append(torch.zeros(hidden_dims[i0 + 1]))

        # output layer weights
        output_w_id = torch.eye(hidden_dims[-1], output_dim)
        if normalize:
            output_w_random = gain_factor * torch.randn(hidden_dims[-1], output_dim) / math.sqrt(hidden_dims[-1])
        else:
            output_w_random = gain_factor * torch.randn(hidden_dims[-1], output_dim)

        output_w = (1 - pert_factor) * output_w_id + pert_factor * output_w_random
        layers.append(output_w)
        nonlinearities.append(self.nonlinearity)
        biases.append(torch.zeros(output_dim))

        super().__init__(layers, biases, nonlinearities)

class SymmetricDenseRandomFF(FeedForward):
    """
    Feedforward net with an equal number of hidden units in each layer. Weights are initialized according to two
    factors: how close to an "identity" matrix the weights are, and a gain factor. Biases are initialized to zero.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, pert_factor: float = 1.,
                 gain_factor: float = 0.4, nonlinearity: typing.Union[str, typing.Callable] = 'relu',
                 normalize: bool = True):

        if isinstance(nonlinearity, typing.Callable):
            self.nonlinearity = nonlinearity
        elif isinstance(nonlinearity, str):
            if nonlinearity == 'tanh' or nonlinearity == 'Tanh':
                self.nonlinearity = torch.tanh
            elif nonlinearity == 'relu' or nonlinearity == 'ReLU':
                def relu(x):
                    return torch.clamp(x, min=9)

                self.nonlinearity = relu
            else:
                raise AttributeError("nonlinearity not recognized.")
        else:
            raise AttributeError("nonlinearity not recognized.")

        layers = []
        nonlinearities = []
        biases = []

        # input weights
        input_w_id = torch.eye(input_dim, hidden_dim)
        if normalize:
            input_w_random = gain_factor * torch.randn(input_dim, hidden_dim) / math.sqrt(input_dim)
        else:
            input_w_random = gain_factor * torch.randn(input_dim, hidden_dim)

        input_w = (1 - pert_factor) * input_w_id + pert_factor * input_w_random
        layers.append(input_w)
        nonlinearities.append(self.nonlinearity)
        biases.append(torch.zeros(hidden_dim))

        # hidden layer weights
        for i0 in range(num_layers - 1):
            hidden_w_id = torch.eye(hidden_dim, hidden_dim)
            if normalize:
                hidden_w_random = gain_factor * torch.randn(hidden_dim, hidden_dim) / math.sqrt(hidden_dim)
            else:
                hidden_w_random = gain_factor * torch.randn(hidden_dim, hidden_dim)

            hidden_w = (1 - pert_factor) * hidden_w_id + pert_factor * hidden_w_random
            layers.append(hidden_w)
            nonlinearities.append(self.nonlinearity)
            biases.append(torch.zeros(hidden_dim))

        # output layer weights
        output_w_id = torch.eye(hidden_dim, output_dim)
        if normalize:
            output_w_random = gain_factor * torch.randn(hidden_dim, output_dim) / math.sqrt(hidden_dim)
        else:
            output_w_random = gain_factor * torch.randn(hidden_dim, output_dim)

        output_w = (1 - pert_factor) * output_w_id + pert_factor * output_w_random
        layers.append(output_w)
        nonlinearities.append(self.nonlinearity)
        biases.append(torch.zeros(output_dim))

        super().__init__(layers, biases, nonlinearities)

class SymmetricFeedForward(nn.Module):
    """
    Class for symmetric feedforward neural network model. Takes a list of pytorch tensors and ties these together into a
    trainable neural network, where the second half of the network is a symmetric copy of the first half.
    """

    def __init__(self, layers: List[torch.Tensor], biases: List[torch.Tensor], nonlinearities: List[Callable],
                 train_biases: bool = False):
        """

        Parameters
        ----------
        layers : List[torch.Tensor]
            A list of torch Tensors that specify the weights of the neural network. This list determines the first half
            of the network; the second half is a symmetric copy of the first half.
        biases : List[torch.Tensor]
        A list of torch Tensors that specify the biases of the neural network. This list determines the first half
            of the network; the second half is a copy of the first half.
        nonlinearities : List[Callable]
            A list of Callables that specify the nonlinearities of the neural network layers. This list determines the
            first half of the network; the second half is a copy of the first half.
        """
        super().__init__()
        self.num_layers_half = len(self.layers)
        self.layers = nn.ParameterList([nn.Parameter(layer, requires_grad=True) for layer in layers])
        self.biases = nn.ParameterList([nn.Parameter(bias, requires_grad=train_biases) for bias in biases])
        self.nonlinearities = nonlinearities

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hid = inputs
        for layer, nonlinearity, bias in zip(self.layers, self.nonlinearities, self.biases[1:]):
            hid = nonlinearity(hid @ layer + bias)
        for i0 in range(self.num_layers_half):
            idx = self.num_layers_half - i0 - 1
            hid = self.nonlinearities[idx](hid @ self.layers[idx].t() + self.biases[idx])
        return hid

    def get_pre_activations(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        hid = inputs
        pre_activations = []
        for layer, nonlinearity, bias in zip(self.layers, self.nonlinearities, self.biases):
            pre_activation = hid @ layer + bias
            hid = nonlinearity(pre_activation)
            pre_activations.append(pre_activation.detach())
        for i0 in range(self.num_layers_half):
            idx = self.num_layers_half - i0 - 1
            pre_activation = hid @ self.layers[idx].t() + self.biases[idx]
            hid = self.nonlinearities[idx](pre_activation)
            pre_activations.append(pre_activation.detach())
        return pre_activations

    def get_post_activation(self, inputs) -> List[torch.Tensor]:
        hid = inputs
        post_activations = []
        for layer, nonlinearity, bias in zip(self.layers, self.nonlinearities, self.biases):
            hid = nonlinearity(hid @ layer + bias)
            post_activations.append(hid.detach())
        for i0 in range(self.num_layers_half):
            idx = self.num_layers_half - i0 - 1
            pre_activation = hid @ self.layers[idx].t() + self.biases[idx]
            hid = self.nonlinearities[idx](pre_activation)
            post_activations.append(hid.detach())
        return post_activations

    def get_activations(self, inputs) -> List[torch.Tensor]:
        hid = inputs
        activations = []
        for layer, nonlinearity, bias in zip(self.layers, self.nonlinearities, self.biases):
            pre_activation = hid @ layer + bias
            hid = nonlinearity(pre_activation)
            activations.append(pre_activation.detach())
            activations.append(hid.detach())
        for i0 in range(self.num_layers_half):
            idx = self.num_layers_half - i0 - 1
            pre_activation = hid @ self.layers[idx].t() + self.biases[idx]
            hid = self.nonlinearities[idx](pre_activation)
            activations.append(pre_activation.detach())
            activations.append(hid.detach())
        return activations

class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN).
    """

    def __init__(self, input_weights: torch.Tensor, recurrent_weights: torch.Tensor, output_weights: torch.Tensor,
                 recurrent_bias: torch.Tensor, output_bias: torch.Tensor,
                 nonlinearity: typing.Union[None, str, typing.Callable],
                 hidden_unit_init: typing.Union[None, str, torch.Tensor] = None,
                 train_input=False, train_recurrent=True, train_output=True, train_recurrent_bias=True,
                 train_output_bias=True):

        super().__init__()

        if isinstance(nonlinearity, typing.Callable):
            self.nonlinearity = nonlinearity
        elif isinstance(nonlinearity, str):
            if nonlinearity == 'tanh' or nonlinearity == 'Tanh':
                self.nonlinearity = torch.tanh
            elif nonlinearity == 'relu' or nonlinearity == 'ReLU':
                def relu(x):
                    return torch.clamp(x, min=9)

                self.nonlinearity = relu
            else:
                raise AttributeError("nonlinearity not recognized.")
        else:
            raise AttributeError("nonlinearity not recognized.")

        if hidden_unit_init is None:
            self.hidden_unit_init = 0
        elif isinstance(hidden_unit_init, torch.Tensor):
            self.hidden_unit_init = hidden_unit_init
        else:
            raise AttributeError("hidden_unit_init option not recognized.")

        if train_input:
            self.Win = nn.Parameter(input_weights, requires_grad=True)
        else:
            self.Win = nn.Parameter(input_weights, requires_grad=False)

        if train_recurrent:
            self.Wrec = torch.Parameter(recurrent_weights, requires_grad=True)
        else:
            self.Wrec = torch.Parameter(recurrent_weights, requires_grad=False)

        if train_output:
            self.Wout = torch.Parameter(output_weights, requires_grad=True)
        else:
            self.Wout = torch.Parameter(output_weights, requires_grad=False)

        if train_recurrent_bias:
            self.brec = torch.Parameter(recurrent_bias, requires_grad=True)
        else:
            self.brec = torch.Parameter(recurrent_bias, requires_grad=False)

        if train_output_bias:
            self.bout = torch.Parameter(output_bias, requires_grad=True)
        else:
            self.bout = torch.Parameter(output_bias, requires_grad=False)

    def forward(self, inputs):
        hid = self.hidden_unit_init
        for i0 in range(inputs.shape[1]):
            hid = self.nonlinearity()
        out = hid @ self.Wout + self.bout
        return out

    def get_pre_activations(self, inputs):
        hid = self.hidden_unit_init
        preactivations = []
        for i0 in range(inputs.shape[1]):
            preactivation = hid @ self.Wrec + inputs[:, i0] @ self.Win + self.brec
            hid = self.nonlinearity(preactivation)
            preactivations.append(preactivation.detach())
        out = hid @ self.Wout + self.bout
        preactivations.append(out.detach())
        return preactivations

    def get_post_activation(self, inputs):
        hid = self.hidden_unit_init
        postactivations = []
        for i0 in range(inputs.shape[1]):
            preactivation = hid @ self.Wrec + inputs[:, i0] @ self.Win + self.brec
            hid = self.nonlinearity(preactivation)
            postactivations.append(hid.detach())
        out = hid @ self.Wout + self.bout
        postactivations.append(out.detach())
        return postactivations

    def get_activations(self, inputs):
        hid = self.hidden_unit_init
        activations = []
        for i0 in range(inputs.shape[1]):
            preactivation = hid @ self.Wrec + inputs[:, i0] @ self.Win + self.brec
            hid = self.nonlinearity(preactivation)
            activations.append(preactivation.detach())
            activations.append(hid.detach())
        out = hid @ self.Wout + self.bout
        activations.append(out.detach())
        return activations

class StaticInputRNN(RNN):

    def forward(self, inputs, num_recurrent_steps):
        hid = self.hidden_unit_init
        preactivation = hid @ self.Wrec + inputs @ self.Win + self.brec
        hid = self.nonlinearity(preactivation)
        for i0 in range(num_recurrent_steps - 1):
            preactivation = hid @ self.Wrec + self.brec
            hid = self.nonlinearity(preactivation)
        out = hid @ self.Wout + self.bout
        return out

    def get_pre_activations(self, inputs, num_recurrent_steps):
        hid = self.hidden_unit_init
        preactivations = []
        preactivation = hid @ self.Wrec + inputs @ self.Win + self.brec
        hid = self.nonlinearity(preactivation)
        preactivations.append(preactivation.detach())
        for i0 in range(num_recurrent_steps - 1):
            preactivation = hid @ self.Wrec + self.brec
            hid = self.nonlinearity(preactivation)
            preactivations.append(preactivation.detach())
        out = hid @ self.Wout + self.bout
        preactivations.append(out.detach())
        return preactivations

    def get_post_activation(self, inputs, num_recurrent_steps):
        hid = self.hidden_unit_init
        postactivations = []
        preactivation = hid @ self.Wrec + inputs @ self.Win + self.brec
        hid = self.nonlinearity(preactivation)
        postactivations.append(hid.detach())
        for i0 in range(num_recurrent_steps - 1):
            preactivation = hid @ self.Wrec + self.brec
            hid = self.nonlinearity(preactivation)
            postactivations.append(hid.detach())
        out = hid @ self.Wout + self.bout
        postactivations.append(out.detach())
        return postactivations

    def get_activations(self, inputs, num_recurrent_steps):
        hid = self.hidden_unit_init
        activations = []
        preactivation = hid @ self.Wrec + inputs @ self.Win + self.brec
        hid = self.nonlinearity(preactivation)
        activations.append(preactivation.detach())
        activations.append(hid.detach())
        for i0 in range(num_recurrent_steps - 1):
            preactivation = hid @ self.Wrec + self.brec
            hid = self.nonlinearity(preactivation)
            activations.append(preactivation.detach())
            activations.append(hid.detach())
        out = hid @ self.Wout + self.bout
        activations.append(out.detach())
        return activations
