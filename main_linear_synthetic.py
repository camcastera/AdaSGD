import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils.objective_funcs as objective_funcs
import utils.postprocessing as postprocessing
import utils.loops as loops

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
def quadratic_forward(x, A, b, minibatch=None):
    minibatch = range(A.shape[0]) if minibatch is None else minibatch
    return 1 / len(minibatch) * torch.sum(1 / 2 * ((A[minibatch] @ x) - b[minibatch])**2, dim=-1)


######
## Define the dataset
pbname = 'quad_synthetic'

N_samples = 200
n = 20


A_quad = torch.randn(N_samples, n)
b_quad = torch.randn(N_samples)


# Below creates a forward function that only takes x and minibatch as input 
oracle_quad = lambda x, minibatch: quadratic_forward(x, A=A_quad, b=b_quad, minibatch=minibatch)
quad_func = objective_funcs.obj_func(oracle_quad) # This is a class of oracle that can compute f or nabla f
######
#Do one Newton step to compute xstar and fstar
xstar = 0 - torch.linalg.solve(1 / N_samples * A_quad.T @ A_quad, quad_func.eval_gradf(torch.zeros(n)))
fstar = quad_func.eval_f(xstar).item()



###### Figure 1
## Run a grid-search and plot the sensitivity to step-size
n_epoch_GS = 100

optim_parameters = {
    'x0' : 3 * torch.randn(n),
    'oracle' : quad_func,
    'N_samples' : N_samples,
    'batchsize' : 32,
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

Dict_best_results = loops.run_best_configs(optim_parameters, list_best_keys)


# Plot
fig, ax = plt.subplots(figsize=(6, 5))


for i, ((key, stepsize), color, zorder) in enumerate(zip(list_best_keys, list_colors, list_zorders)):
    list_f = np.concatenate(([quad_func.eval_f(optim_parameters['x0'])], Dict_best_results[(key, stepsize)][1]))
    ax.plot(np.array(list_f) - fstar, color=color, lw=2.5, label=key + r' $\lambda_0$ = ' + str(stepsize), zorder=zorder)
ax.set_xlabel(r'number of epochs') ; ax.set_ylabel(r'$f(x_k) - f^\star$')
ax.set_yscale('log')



filename = 'Figures/' + pbname +'_run.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)