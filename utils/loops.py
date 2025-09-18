import utils.optimization_algorithms as optimizers
from tqdm.auto import tqdm

def run_all_algorithms(optim_parameters, list_keys, list_stepsizes):

    # Extract parameters
    x0 = optim_parameters['x0']
    oracle = optim_parameters['oracle']
    N_samples = optim_parameters['N_samples']
    batchsize = optim_parameters['batchsize']
    n_epochs = optim_parameters['n_epochs']
    trainloader = optim_parameters['trainloader']

    # Run all algorithms in all configurations
    Dict_results = {}
    
    progbar = tqdm(list_stepsizes, leave=False)
    for stepsize in progbar:
        for key in list_keys:
            progbar.set_description('stepsize: ' +str(stepsize)+', algo: '+str(key))
            if key == 'SGD, no decay':
                list_f, list_total_loss, _ = optimizers.SGD(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, decay=False)
            elif key == 'SGD, decay':
                list_f, list_total_loss, _ = optimizers.SGD(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, decay=True)
            elif key == 'AdaSGD, no decay':
                list_f, list_total_loss, _ = optimizers.Ada_SGD(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, decay=None)
            elif key == 'AdaSGD, decay v1':
                list_f, list_total_loss, _ = optimizers.Ada_SGD(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, decay='decay_v1')
            elif key == 'AdaSGD, decay v2':
                list_f, list_total_loss, _ = optimizers.Ada_SGD(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, decay='decay_v2')
            elif key == 'AdaSGD_MM, biased':
                list_f, list_total_loss, _ = optimizers.Ada_SGD_MM(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, variant='biased')
            elif key == 'AdaSGD_MM, unbiased':
                list_f, list_total_loss, _ = optimizers.Ada_SGD_MM(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, variant='unbiased')
            Dict_results[(key, stepsize)] = (list_f, list_total_loss)

    return Dict_results


def run_best_configs(optim_parameters, list_best_keys):

    # Extract parameters
    x0 = optim_parameters['x0']
    oracle = optim_parameters['oracle']
    N_samples = optim_parameters['N_samples']
    batchsize = optim_parameters['batchsize']
    n_epochs = optim_parameters['n_epochs']
    trainloader = optim_parameters['trainloader']

    # Run all algorithms in all configurations
    Dict_results = {}

    for key, stepsize in list_best_keys:
        if key == 'SGD, no decay':
            list_f, list_total_loss, list_stepsizes = optimizers.SGD(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, decay=False, progbar=True)
        elif key == 'SGD, decay':
            list_f, list_total_loss, list_stepsizes = optimizers.SGD(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, decay=True, progbar=True)
        elif key == 'AdaSGD, no decay':
            list_f, list_total_loss, list_stepsizes = optimizers.Ada_SGD(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, decay=None, progbar=True)
        elif key == 'AdaSGD, decay v1':
            list_f, list_total_loss, list_stepsizes = optimizers.Ada_SGD(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, decay='decay_v1', progbar=True)
        elif key == 'AdaSGD, decay v2':
            list_f, list_total_loss, list_stepsizes = optimizers.Ada_SGD(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, decay='decay_v2', progbar=True)
        elif key == 'AdaSGD_MM, biased':
            list_f, list_total_loss, list_stepsizes = optimizers.Ada_SGD_MM(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, variant='biased', progbar=True)
        elif key == 'AdaSGD_MM, unbiased':
            list_f, list_total_loss, list_stepsizes = optimizers.Ada_SGD_MM(x0, oracle=oracle, base_stepsize=stepsize, batchsize=batchsize, N_samples=N_samples, n_epochs=n_epochs, trainloader=trainloader, variant='unbiased', progbar=True)
        Dict_results[(key, stepsize)] = (list_f, list_total_loss, list_stepsizes)

    return Dict_results