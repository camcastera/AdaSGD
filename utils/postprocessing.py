from torch import nn
from torch import is_tensor



def compute_final_average(Dict_results, oracle, list_keys, list_stepsizes):
    Dict_last_avg = {}
    for keys, result, in Dict_results.items():
        list_total_loss = result[1]
        Dict_last_avg[keys] = list_total_loss[-1] #last full batch loss
        if Dict_last_avg[keys] != Dict_last_avg[keys]: #check NaNs
            Dict_last_avg[keys] = 5e5
    return Dict_last_avg


def select_best_stepsize(Dict_final_avg, list_keys, list_stepsizes):
    list_best_keys = []
    for key in list_keys:
        best_stepsize = 1e-10
        best_val = 1e100
        for stepsize in list_stepsizes:
            val_f  = Dict_final_avg[(key, stepsize)]
            if val_f < best_val:
                best_val = val_f
                best_stepsize = stepsize
        list_best_keys.append((key, best_stepsize))
    return list_best_keys