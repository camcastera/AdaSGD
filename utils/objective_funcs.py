import torch
from torch import nn


def zero_grad(model): #zero gradients of NNs
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    pass

# This is the generic class that builds oracles based on implementations of forward passes
class obj_func:
    def __init__(self, oracle_f, N_samples=1):
        self.oracle_f = oracle_f
        self.N_samples = N_samples


    def eval_f(self, x, minibatch=None):
        return self.oracle_f(x, minibatch=minibatch)


    def eval_gradf(self, x, minibatch=None):
        if issubclass(x.__class__, nn.Module): #if x is actually a neural network, things are handled a little differently
            zero_grad(x)
            loss = self.eval_f(x, minibatch=minibatch)
            loss.backward()
            grad = [p.grad for p in x.parameters()]
        else:
            y = x.clone().detach().requires_grad_(True)
            loss = self.eval_f(y, minibatch=minibatch)
            #loss.backward()
            grad = torch.autograd.grad(torch.sum(loss, dim=0), inputs=y)#, retain_graph=True)
            grad = grad[0] 

        return grad
    
    def eval_Hess_times_vec(self, x, vec, minibatch=None):
        if issubclass(x.__class__, nn.Module): #if x is actually a neural network, things are handled a little differently
            raise TypeError("Neural Net not implemented yet") 
        else:
            hess_times_vec = torch.autograd.functional.hvp(self.eval_f, inputs=x, v=vec, create_graph=False, strict=False)[1]
        return hess_times_vec