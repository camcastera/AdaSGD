import numpy as np
import torch
from torch import nn
from tqdm.notebook import tqdm

def SGD(x0, oracle, base_stepsize, batchsize, N_samples, n_epochs, decay=False, progbar=False, trainloader=None, evaluate_f=False):
    
    isNN_training = issubclass(x0.__class__, nn.Module)#flag used because training NNs requires a slightly different implementation at many points

    if isNN_training: #if x is a neural network:
        x = x0
        x.load_state_dict(torch.load(x.state_dict_path, weights_only=True)) #load initial weights
    else:
        x = x0.clone().detach()

    list_values = [] ; list_total_loss = [] ; list_stepsizes = []
    
    if trainloader is not None:
        iter_per_epoch = len(trainloader)
    else:
        iter_per_epoch = int(N_samples / batchsize)
        trainloader = range(iter_per_epoch)

    if progbar is True:
        iterator = tqdm(range(n_epochs), leave=False)
        iterator.set_description('   SGD  , stepsize: ' + str(base_stepsize))
    else:
        iterator = range(n_epochs)

    full_batch = range(N_samples) # to avoid recreating a range to draw from at each iter

    for epoch in iterator:
        for k, minibatch in enumerate(trainloader): #this is just a trick to handle trainloader and normal loops at the same time
            #draw minibatch when the iterator does not load data directly
            if not issubclass(trainloader.__class__, torch.utils.data.DataLoader):
                minibatch = np.random.choice(full_batch, size=batchsize, replace=False) #draw minibatch
                
            total_iter = iter_per_epoch * epoch + k
            if evaluate_f is True:
                list_values.append(oracle.eval_f(x, minibatch=minibatch).item())
            grad_sto = oracle.eval_gradf(x, minibatch=minibatch)
            if decay is True:
                stepsize = base_stepsize / (total_iter + 1)**0.51
            else:
                stepsize = base_stepsize
            list_stepsizes.append(stepsize)

            if isNN_training: #if x is a neural network
                with torch.no_grad():
                    [p.add_(- stepsize * p.grad) for p in x.parameters()]
            else:
                x = x - stepsize * grad_sto      

        # At the end of each epoch, compute the total loss
        if not isNN_training:
            with torch.no_grad():
                list_total_loss.append(oracle.eval_f(x, minibatch=None))
            
    #post training
    if isNN_training:
        last_train_loss = 0.
        #Exhaust the dataloader just to be sure to do a full batch
        for _, minibatch in enumerate(trainloader):
            pass
        #Then actually compute the loss over the full batch
        for _, minibatch in enumerate(trainloader):
            last_train_loss += oracle.eval_f(x, minibatch=minibatch).item()
        last_train_loss /= len(trainloader) #normalize by number of batchsize (each minibatch already normalizes)
        return list_values, last_train_loss, list_stepsizes

    return list_values, list_total_loss, list_stepsizes



##################################################################
def Ada_SGD(x0, oracle, base_stepsize, batchsize, N_samples, n_epochs, decay=None, progbar=False, trainloader=None, evaluate_f=False):


    isNN_training = issubclass(x0.__class__, nn.Module)#flag used because training NNs requires a slightly different implementation at many points

    if isNN_training: #if x is a neural network:
        x = x0
        x.load_state_dict(torch.load(x.state_dict_path, weights_only=True)) #load initial weights
    else:
        x = x0.clone().detach()

    list_values = [] ; list_total_loss = [] ; list_stepsizes = []
    lmbda = torch.tensor(base_stepsize)

    if trainloader is not None:
        iter_per_epoch = len(trainloader)
    else:
        iter_per_epoch = int(N_samples / batchsize)
        trainloader = range(iter_per_epoch)

    if progbar is True:
        iterator = tqdm(range(n_epochs), leave=False)
        iterator.set_description('AdaSGD  , stepsize: ' + str(base_stepsize))
    else:
        iterator = range(n_epochs)

    full_batch = range(N_samples) # to avoid recreating a range to draw from at each iter

    for epoch in iterator:
        for k, minibatch in enumerate(trainloader): #this is just a trick to handle trainloader and normal loops at the same time
            #draw minibatch when the iterator does not load data directly
            if not issubclass(trainloader.__class__, torch.utils.data.DataLoader):
                minibatch = np.random.choice(full_batch, size=batchsize, replace=False) #draw minibatch
        
            if evaluate_f is True:
                list_values.append(oracle.eval_f(x, minibatch=minibatch).item())
            grad_sto = oracle.eval_gradf(x, minibatch=minibatch)
            
            total_iter = iter_per_epoch * epoch + k

            if total_iter >= 1: #after 2 iterations start adapting automatically
                extra_grad_sto = oracle.eval_gradf(x, minibatch=previous_minibatch) # nabla f_{\xi_{k-1}}(x_k)
                if isNN_training:
                    with torch.no_grad():
                        square_dx = torch.tensor(0.) ; square_dg = torch.tensor(0.)
                        [square_dx.add_( torch.sum((p.ravel() - prev_p.ravel())**2, dim=-1)) for (p, prev_p) in zip(x.parameters(), previous_x)]
                        [square_dg.add_(torch.sum((extra_g.ravel() - prev_g.ravel())**2, dim=-1)) for (extra_g, prev_g) in zip(extra_grad_sto, previous_grad_sto)]
                        cond1 = torch.sqrt(square_dx) / torch.sqrt(8 * square_dg)
                else:
                    cond1 = torch.sqrt(torch.sum((x - previous_x)**2, dim=-1)) / torch.sqrt(8 * torch.sum((extra_grad_sto - previous_grad_sto)**2, dim=-1))

                theta = lmbda_km1 / lmbda_km2 if total_iter >= 2 else torch.tensor(1e100) #theta is infinity at first 2 iterations
                if decay == 'decay_v2':
                    cond2 = lmbda_km1 * torch.sqrt(1. + (1. - (total_iter + 1)**-0.5001) * theta)
                else:
                    cond2 = lmbda_km1 * torch.sqrt(1. + theta)


                if decay in ['decay_v1', 'decay_v2']:
                    cond1 = cond1 / (total_iter + 1)**0.5001

            
            if total_iter >= 2:
                stepsize = torch.fmin(cond1, cond2)
            elif total_iter == 1:
                stepsize = cond1
            else:
                stepsize = base_stepsize
            list_stepsizes.append(stepsize)


            #store variables for next iter
            previous_minibatch = minibatch
            if isNN_training: #if x is a neural network:
                previous_x = [p.clone().detach() for p in x.parameters()]
                previous_grad_sto = [g.clone().detach() for g in grad_sto]
            else:
                previous_x = x.clone().detach()
                previous_grad_sto = grad_sto.clone().detach()
            if total_iter>=1:
                lmbda_km2 = lmbda_km1.clone().detach() if torch.is_tensor(lmbda_km1) else torch.tensor(lmbda_km1) # stepsize from 2 iters ago
            lmbda_km1 = stepsize.clone().detach() if torch.is_tensor(stepsize) else stepsize#previous stepsize

            #update x
            if isNN_training: #if x is a neural network
                with torch.no_grad():
                    [p.add_(- stepsize * p.grad) for p in x.parameters()]
            else:
                x = x - stepsize * grad_sto  

        # At the end of each epoch, compute the total loss
        if not isNN_training:
            with torch.no_grad():
                list_total_loss.append(oracle.eval_f(x, minibatch=None))


    # post training
    if isNN_training:
        last_train_loss = 0.
        #Exhaust the dataloader just to be sure to do a full batch
        for _, minibatch in enumerate(trainloader):
            pass
        #Then actually compute the loss over the full batch
        for _, minibatch in enumerate(trainloader):
            last_train_loss += oracle.eval_f(x, minibatch=minibatch).item()
        last_train_loss /= len(trainloader) #normalize by number of batchsize (each minibatch already normalizes)
        return list_values, last_train_loss, list_stepsizes
    
        
    return list_values, list_total_loss, list_stepsizes




################################################################
def Ada_SGD_MM(x0, oracle, base_stepsize, batchsize, N_samples, n_epochs, variant='biased', progbar=False, trainloader=None, evaluate_f=False):

    
    isNN_training = issubclass(x0.__class__, nn.Module)#flag used because training NNs requires a slightly different implementation at many points

    if isNN_training: #if x is a neural network:
        x = x0
        x.load_state_dict(torch.load(x.state_dict_path, weights_only=True)) #load initial weights
    else:
        x = x0.clone().detach()

    list_values = [] ; list_total_loss = [] ; list_stepsizes = []
    lmbda = torch.tensor(base_stepsize)

    if trainloader is not None:
        iter_per_epoch = len(trainloader)
    else:
        iter_per_epoch = int(N_samples / batchsize)
        trainloader = range(iter_per_epoch)

    
    if progbar is True:
        iterator = tqdm(range(n_epochs), leave=False)
        iterator.set_description('AdaSGD MM, stepsize: ' + str(base_stepsize))
    else:
        iterator = range(n_epochs)

    full_batch = range(N_samples) # to avoid recreating a range to draw from at each iter

    for epoch in iterator:
        for k, minibatch in enumerate(trainloader): #this is just a trick to handle trainloader and normal loops at the same time
            #draw minibatch when the iterator does not load data directly
            if not issubclass(trainloader.__class__, torch.utils.data.DataLoader):
                minibatch = np.random.choice(full_batch, size=batchsize, replace=False) #draw minibatch

            if evaluate_f is True:
                list_values.append(oracle.eval_f(x, minibatch=minibatch).item())
            grad_sto = oracle.eval_gradf(x, minibatch=minibatch)
            
            total_iter = iter_per_epoch * epoch + k

            if total_iter >= 1: #after 2 iterations start adapting automatically
                
                
                
                if variant=='biased':
                    extra_grad_sto = oracle.eval_gradf(previous_x, minibatch=minibatch) # nabla f_{\xi_{k-1}}(x_k)
                    if isNN_training:
                        with torch.no_grad():
                            square_dx = torch.tensor(0.) ; square_dg = torch.tensor(0.)
                            [square_dx.add_( torch.sum((p.ravel() - prev_p.ravel())**2, dim=-1)) for (p, prev_p) in zip(x.parameters(), previous_x)]
                            [square_dg.add_(torch.sum((extra_g.ravel() - g.ravel())**2, dim=-1)) for (extra_g, g) in zip(extra_grad_sto, grad_sto)]
                            cond1 = torch.sqrt(square_dx) / torch.sqrt(square_dg)
                    else:
                        cond1 = torch.sqrt(torch.sum((x - previous_x)**2, dim=-1)) / torch.sqrt(torch.sum((grad_sto - extra_grad_sto)**2, dim=-1))
                elif variant=='unbiased':
                    extra_minibatch = np.random.choice(full_batch, size=batchsize, replace=False) #draw minibatch
                    extra_grad_sto_1 = oracle.eval_gradf(x, minibatch=extra_minibatch) # nabla f_{\xi_{k-1}}(x_k)
                    extra_grad_sto_2 = oracle.eval_gradf(previous_x, minibatch=extra_minibatch) # nabla f_{\xi_{k-1}}(x_k)
                    if isNN_training:
                        with torch.no_grad():
                            square_dx = torch.tensor(0.) ; square_dg = torch.tensor(0.)
                            [square_dx.add_( torch.sum((p.ravel() - prev_p.ravel())**2, dim=-1)) for (p, prev_p) in zip(x.parameters(), previous_x)]
                            [square_dg.add_(torch.sum((extra_g1.ravel() - extra_g2.ravel())**2, dim=-1)) for (extra_g1, extra_g2) in zip(extra_grad_sto1, extra_grad_sto2)]
                            cond1 = torch.sqrt(square_dx) / torch.sqrt(square_dg)
                    else:
                        cond1 = torch.sqrt(torch.sum((x - previous_x)**2, dim=-1)) / torch.sqrt(torch.sum((extra_grad_sto_1 - extra_grad_sto_2)**2, dim=-1))
                
                cond1 = base_stepsize * cond1 

                theta = lmbda_km1 / lmbda_km2 if total_iter >= 2 else torch.tensor(1e100) #theta is infinity at first 2 iterations
                cond2 = lmbda_km1 * torch.sqrt(1. + theta)
            

            

            if total_iter >= 2:
                stepsize = torch.fmin(cond1, cond2)
            elif total_iter == 1:
                stepsize = cond1
            else:
                stepsize = base_stepsize
            list_stepsizes.append(stepsize)


            #store variables for next iter
            previous_minibatch = minibatch
            if isNN_training: #if x is a neural network:
                previous_x = [p.clone().detach() for p in x.parameters()]
            else:
                previous_x = x.clone().detach()
            if total_iter>=1:
                lmbda_km2 = lmbda_km1.clone().detach() if torch.is_tensor(lmbda_km1) else torch.tensor(lmbda_km1) # stepsize from 2 iters ago
            lmbda_km1 = stepsize.clone().detach() if torch.is_tensor(stepsize) else stepsize#previous stepsize

            #update x
            if isNN_training: #if x is a neural network
                with torch.no_grad():
                    [p.add_(- stepsize * p.grad) for p in x.parameters()]
            else:
                x = x - stepsize * grad_sto  
        
        # At the end of each epoch, compute the total loss
        if not isNN_training:
            with torch.no_grad():
                list_total_loss.append(oracle.eval_f(x, minibatch=None))

    #Post training
    if isNN_training:
        last_train_loss = 0.
        #Exhaust the dataloader just to be sure to do a full batch
        for _, minibatch in enumerate(trainloader):
            pass
        #Then actually compute the loss over the full batch
        for _, minibatch in enumerate(trainloader):
            last_train_loss += oracle.eval_f(x, minibatch=minibatch).item()
        last_train_loss /= len(trainloader) #normalize by number of batchsize (each minibatch already normalizes)
        return list_values, last_train_loss, list_stepsizes
        
    return list_values, list_total_loss, list_stepsizes