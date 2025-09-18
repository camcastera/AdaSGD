import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils.objective_funcs as objective_funcs
import utils.postprocessing as postprocessing
import utils.loops as loops
from sklearn.datasets import load_svmlight_file


# Below avoids problems with LateX in papers
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

torch.manual_seed(987654321)


# List of algorithms and grid of stepsizes
list_keys = ['SGD, no decay', 'SGD, decay', 'AdaSGD, no decay', 'AdaSGD, decay v1', 'AdaSGD, decay v2', 'AdaSGD_MM, biased', 'AdaSGD_MM, unbiased']
start = -4 ; stop = 2 ; num = 2 * (stop - start) + 1
list_stepsizes = 10**np.linspace(start, stop, num=num) # log-scale evenly-spaced grid
list_colors = ['limegreen', 'orange', 'dodgerblue', 'blue', 'cyan', 'magenta', 'pink']
list_zorders = [5, 5 , 6, 7, 7, 3, 3]


######


#define the general loss function
def logistic_forward(X, A, b, minibatch=None):
    minibatch = range(A.shape[0]) if minibatch is None else minibatch
    correl = (A[minibatch] @ X).squeeze()  #Linear model times x in logistic regression
    if torch.sum(b == 0) > 0: #check if b contains zeros
        warnings.warn('Warning: this implementation expects the b in {-1,1} formulation')

    loglike = torch.log(1. + torch.exp(-b[minibatch] * correl)) #vector of all the log likelihood b is binary \in {0,1}
    g = 1 / len(minibatch) * torch.sum(loglike, dim=-1) #+ regul #returns one value per data sample
    return g


######
## Define the dataset
pbname = 'logreg_w8a'

from sklearn.datasets import load_svmlight_file
A_w8a, b_w8a = load_svmlight_file('datasets/w8a')
A_w8a = torch.tensor(A_w8a.todense()).float()
b_w8a = torch.tensor(b_w8a).float()
b_w8a[b_w8a == 2] = -1.


N_samples, n = A_w8a.shape[-2:]

oracle_logistic = lambda x, minibatch: logistic_forward(x, A_w8a, b_w8a, minibatch=minibatch)

log_func = objective_funcs.obj_func(oracle_logistic)
######
#Load fstar
fstar = np.load('datasets/fstar_' + pbname + '.npy')



###### Figure 1
## Run a grid-search and plot the sensitivity to step-size
n_epoch_GS = 10

optim_parameters = {
    'x0' : 3 * torch.randn(n),
    'oracle' : log_func,
    'N_samples' : N_samples,
    'batchsize' : 309,
    'n_epochs' : n_epoch_GS,
    'trainloader': None
}

# Grid-search
Dict_results = loops.run_all_algorithms(optim_parameters, list_keys, list_stepsizes)



# Plot
fig, ax = plt.subplots(figsize=(6, 5))

avg_every = N_samples
Dict_final_avg = postprocessing.compute_final_average(Dict_results, optim_parameters['oracle'], list_keys, list_stepsizes)

for key, color, zorder in zip(list_keys, list_colors, list_zorders):
    list_best_avg = []
    for stepsize in list_stepsizes:
        list_best_avg.append(Dict_final_avg[(key, stepsize)])
    ax.plot(list_stepsizes, np.array(list_best_avg) - fstar, color=color, lw=2.5, label=key, marker='o', zorder=zorder)


ax.set_xlabel(r'initial step-size $\lambda_0$') ; ax.set_ylabel(r'$f(x_k) - f^\star$')
ax.set_xscale('log') ; ax.set_yscale('log')
ax.set_ylim(ymax=1e2)


# ax.legend()


filename = 'Figures/' + pbname +'_lambda0.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)



## Figure 2

## Select the best config and run again all algorithms for more iterations
list_best_keys = postprocessing.select_best_stepsize(Dict_final_avg, list_keys, list_stepsizes)
for i, key in enumerate(list_best_keys):
    if 'AdaSGD,' in key[0]:
        list_best_keys[i] = (key[0], np.float64(1e-3)) # set the step-size to default value (no GS)

optim_parameters['n_epochs'] = 100
Dict_best_results = loops.run_best_configs(optim_parameters, list_best_keys)


# Plot
fig, ax = plt.subplots(figsize=(6, 5))


for i, ((key, stepsize), color, zorder) in enumerate(zip(list_best_keys, list_colors, list_zorders)):
    list_f = np.concatenate(([log_func.eval_f(optim_parameters['x0'])], Dict_best_results[(key, stepsize)][1]))
    ax.plot(np.array(list_f) - fstar, color=color, lw=2.5, label=key + r' $\lambda_0$ = ' + str(stepsize), zorder=zorder)
ax.set_xlabel(r'number of epochs') ; ax.set_ylabel(r'$f(x_k) - f^\star$')
ax.set_yscale('log')



filename = 'Figures/' + pbname +'_run.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)