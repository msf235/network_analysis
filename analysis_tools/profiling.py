import torch as th
from torch import nn
import models
import math
import time
import timeit

class RNN_builtin(nn.Module):
    def __init__(self, win, wrec, wout, brec, bout):
        super(RNN_builtin, self).__init__()
        s = win.shape
        self.rnn = nn.RNN(s[0], s[1], 1)
        self.rnn.weight_ih_l0.data = win.t().clone()
        self.rnn.weight_hh_l0.data = wrec.t().clone()
        self.wout = nn.Parameter(wout.clone(), requires_grad=True)
        self.rnn.bias_ih_l0.data[:] = 0
        self.rnn.bias_ih_l0.requires_grad_(False)
        self.rnn.bias_hh_l0.data = brec.clone()
        self.bout = nn.Parameter(bout.clone(), requires_grad=True)

    def forward(self, inputs):
        hids = self.rnn(inputs)
        # print(hids[0][:,-1].shape)
        # print(self.wout.shape)
        # print((hids[0][:,-1] @ self.wout).shape)
        return hids[0][:,-1] @ self.wout + self.bout


# def profile():
if __name__ == '__main__':
    d = 10
    N = 5
    c = 2
    b = 10
    T = 6
    lr = .005
    win = th.randn(d, N) / math.sqrt(10)
    wrec = th.randn(N,N) / math.sqrt(5)
    wout = th.randn(N,c)
    brec = th.zeros(N)
    bout = th.zeros(c)
    num_batches = 2000

    def train_model(model, optimizer):
        bhalf = int(round(b/2))
        for i0 in range(num_batches):
            optimizer.zero_grad()
            model.zero_grad()

            inputs = th.zeros(b, T, d)
            inputs_0 = 0.1*th.randn(bhalf, T, d) - 0.2
            inputs_0[:, 1:] = 0
            inputs_1 = 0.1*th.randn(bhalf, T, d) + 0.2
            inputs_1[:, 1:] = 0
            inputs[:bhalf] = inputs_0
            inputs[bhalf:] = inputs_1

            targets = th.zeros(b, 2)
            targets[:bhalf] = th.Tensor([1, 0])
            targets[bhalf:] = th.Tensor([0, 1])

            out = model(inputs)

            loss = th.mean(th.norm(out - targets)**2)
            # print(loss.item())
            loss.backward()
            optimizer.step()

    # rnn1 = th.jit.script(RNN_builtin(win, wrec, wout, brec, bout))
    # optimizer1 = th.optim.SGD(filter(lambda p: p.requires_grad, rnn1.parameters()), lr=lr)

    # rnn2 = th.jit.script(models.RNN(win, wrec, wout, brec, bout, nonlinearity='tanh', train_input=True))
    # rnn2 = th.jit.trace(models.RNN(win, wrec, wout, brec, bout, nonlinearity='tanh', train_input=True),
    #                     th.zeros(b, T, d))
    rnn2 = models.RNN(win, wrec, wout, brec, bout, nonlinearity='tanh', train_input=True)
    optimizer2 = th.optim.SGD(filter(lambda p: p.requires_grad, rnn2.parameters()), lr=lr)

    # tic = time.time()
    # train_model(rnn1, optimizer1)
    # toc = time.time()
    # print(toc-tic)

    tic = time.time()
    train_model(rnn2, optimizer2)
    toc = time.time()
    print(toc-tic)



    # print


# if __name__ == '__main__':
#     print(timeit.timeit("profile()", setup="from __main__ import profile"))
