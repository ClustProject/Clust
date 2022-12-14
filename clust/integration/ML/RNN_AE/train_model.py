import copy
import numpy as np

import torch
import torch.nn as nn


def train_model(model, train_dataloader, parameter):
    """
    Args:
        model (model): model 
        train_dataloader (DataLoader): dataloader for training
        parameter (dictionary): config
        
    Returns:
        trined model: model - trined model
    
    """
    n_epochs, lr, device = parameter['num_epochs'], parameter['learning_rate'], parameter['device']
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss(reduction='sum')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000
    history = []
    for epoch in range(1, n_epochs + 1):
        print("epoch : ", epoch)
        model = model.train()
        losses = []
        for x in train_dataloader:
            optimizer.zero_grad()

            x = x[0].to(device)
            output = model(x)

            loss = criterion(output, x)
            losses.append(loss.item())
            
            
            loss.backward()
            optimizer.step()
        
        epoch_loss = np.mean(losses)
        history.append(epoch_loss)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        if epoch_loss > 100000:
            print("+++++++++++++++++++")
            print("epoch_loss : " , epoch_loss)
            print("best_loss : ", best_loss)
    print("=======================")
    model.load_state_dict(best_model_wts)
    return model, history


def get_representation(model, dataloader, parameter):
    """
    Args:
        model (model): model 
        dataloader (DataLoader): dataloader for inference
        parameter (dictionary): config
        
    Returns:
        np.array: result - represented vectors
    
    """
    device = parameter['device']
    model = model.eval()

    result = []
    with torch.no_grad():
        for x in dataloader:
            x = x = x[0].to(device)
            
            hidden = model.encoder(x)
            last_hidden = hidden[:, -1, :]
            result.append(last_hidden.cpu().numpy())
    result = np.concatenate(np.array(result), 0)
    return result
