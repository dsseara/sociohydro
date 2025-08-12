from census_dataset import CensusDataset
from scipy.interpolate import interp1d
import torch
import numpy as np

import os
import glob
import yaml
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

def get_val_error(model, county, dataset_kwargs):
    """ Return validation error (total MSE across simulation)
        as well as relative validation error compared to no-dynamics baseline
        As in train_census_nn.py, convert everything to the same units regardless of inputs
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = CensusDataset(county, **dataset_kwargs).validate()
    wb1980 = dataset.wb(1980)
    wb2020 = dataset.wb(2020)

    # Neural network prediction
    batch = dataset[0]
    wb = batch['wb'].to(device)
    housing = batch['housing'][None, None].to(device)
    housing[housing.isnan()] = 0.
    wb[wb.isnan()] = 0.
    with torch.no_grad():
        wbNN = model.simulate(wb[0:1], n_steps=40, dt=1, housing=housing)[0,-1].cpu().numpy()

    if dataset_kwargs['use_fill_frac']:
        err_func = lambda wb0, wb1: np.power(wb0 * dataset.housing - wb1 * dataset.housing, 2).sum(0)[dataset.mask]
    elif dataset_kwargs['use_max_scaling']:
        err_func = lambda wb0, wb1: np.power(wb0 - wb1, 2).sum(0)[dataset.mask] * dataset.housing[dataset.mask].max()**2
    else:
        err_func = lambda wb0, wb1: np.power(wb0 - wb1, 2).sum(0)[dataset.mask]

    mse_NN = np.mean(err_func(wbNN, wb2020))
    mse_No = np.mean(err_func(wb1980, wb2020))
    return {
        'val_err': mse_NN,
        'rel_val_err': mse_NN / mse_No
    }

def get_val_errors(model, counties, dataset_kwargs):
    """ Return validation error (total MSE across validation counties) 
        and also compute relative validation error compared to no-dynamics
        baseline
    """
    ret = {
        'val_err': 0., # Total error over all counties
        'rel_val_err': 0., # Average relative error (counties are equal weight)
    }

    print(f'Computing validation errors over {counties}', flush=True)

    for county in counties:
        cerr = get_val_error(model, county, dataset_kwargs)
        ret['val_err'] += cerr['val_err']
        ret['rel_val_err'] += cerr['rel_val_err'] / len(counties)
    
    return ret

def make_predictions_plot(model, county, dataset_kwargs):
    dataset = CensusDataset(county, **dataset_kwargs).validate()
    wb1980 = dataset.wb(1980)
    wb2020 = dataset.wb(2020)

    # Neural network prediction
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch = dataset[0]
    wb = batch['wb'].to(device)
    housing = batch['housing'][None, None].to(device)
    housing[housing.isnan()] = 0.
    wb[wb.isnan()] = 0.
    with torch.no_grad():
        wbNN = model.simulate(wb[0:1], n_steps=40, dt=1, housing=housing)[0,-1].cpu().numpy()

    # Spline extrapolation
    spline = interp1d([1980, 1990], np.stack([dataset.wb(1980), dataset.wb(1990)]), axis=0, fill_value='extrapolate')
    wbSpline = spline(2020)

    # Make sure we have a common evaluation function to compare models with different input scalings
    # Convert all predictions to units of [1 / dam^2] before calculating error
    if dataset_kwargs['use_fill_frac']:
        err_func = lambda wb0, wb1: np.power(wb0 * dataset.housing - wb1 * dataset.housing, 2).sum(0)[dataset.mask]
    elif dataset_kwargs['use_max_scaling']:
        err_func = lambda wb0, wb1: np.power(wb0 - wb1, 2).sum(0)[dataset.mask] * dataset.housing[dataset.mask].max()**2
    else:
        err_func = lambda wb0, wb1: np.power(wb0 - wb1, 2).sum(0)[dataset.mask]

    mse_NN = np.mean(err_func(wbNN, wb2020))
    mse_Sp = np.mean(err_func(wbSpline, wb2020))
    mse_No = np.mean(err_func(wb1980, wb2020))

    print(f'For county = {county}')
    print(f'-----------------------------------------------')
    print(f'        Model         |   MSE     |  Rel. MSE |')
    print(f'-----------------------------------------------')
    print(f'Neural Network.       |  {mse_NN:-8.3g} | {mse_NN/mse_No:-8.3f}  |')
    print(f'Spline extrapolation  |  {mse_Sp:-8.3g} | {mse_Sp/mse_No:-8.3f}  |')
    print(f'No dynamics           |  {mse_No:-8.3g} | {mse_No/mse_No:-8.3f}  |')

    alpha = np.ones(dataset.mask.shape)
    alpha[~dataset.mask] = 0.

    def plot(column, dataset, wb):
        vmax = dataset.housing[dataset.mask].max()
        prop = (wb[0] - wb[1]) / wb.sum(0)
        pc0 = column[0].pcolormesh(dataset.x, dataset.y, prop, vmin=-1, vmax=1, cmap='bwr_r', alpha=alpha)
        pc1 = column[1].pcolormesh(dataset.x, dataset.y,  wb[0], vmin=0, vmax=dataset.vmax(), cmap='Blues', alpha=alpha)
        pc2 = column[2].pcolormesh(dataset.x, dataset.y,  wb[1], vmin=0, vmax=dataset.vmax(), cmap='Reds', alpha=alpha)
        return pc0, pc1, pc2

    fig, ax = plt.subplots(3, 4, dpi=200)

    plot(ax[:,0], dataset, wb1980)
    plot(ax[:,1], dataset, wb2020)
    plot(ax[:,2], dataset, wbNN)
    pc0, pc1, pc2 = plot(ax[:,3], dataset, wbSpline)

    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 8

    if dataset_kwargs['use_fill_frac']:
        clabel='Fill Fraction'
        ticks=[0,1]
    elif dataset_kwargs['use_max_scaling']:
        clabel='Relative\npopulation density'
        ticks=[0,1]
    else:
        clabel='Population density\n[$1/$dam$^2$]'
        ticks=None

    fig.colorbar(pc0, ax=ax[0,:], ticks=[-1,1]).set_ticklabels(['All\nblack', 'All\nwhite'], 
        rotation=90, verticalalignment='center', multialignment='center')
    fig.colorbar(pc1, ax=ax[1,:], ticks=ticks, label=clabel)
    fig.colorbar(pc2, ax=ax[2,:], ticks=ticks, label=clabel)

    ax[0,0].set_title('1980\ncensus data')
    ax[0,1].set_title('2020\ncensus data')
    ax[0,2].set_title('2020\nNN prediction')
    ax[0,3].set_title('2020\nSpline extrapolation')

    ax[0,0].set_ylabel('Resident\nproportion')
    ax[1,0].set_ylabel('White\noccupation')
    ax[2,0].set_ylabel('Black\noccupation')

    for a in ax.flatten():
        a.set(xticks=[], yticks=[], aspect='equal')

    return mse_NN

def report_model(path):
    info = torch.load(f'{path}/model_weight.ckpt', map_location='cpu')

    with open(f'{path}/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    ret = {
        'path': os.path.basename(path),
        'id': os.path.basename(path).replace('_all_counties', ''),
        'val_err': info['val_err'],
        'rel_val_err': info['rel_val_err'],
        'val_loss': info['val_loss'],
        'train_loss': info['train_loss'],
        'val_tmax': config['dataset']['val_tmax'],
        'num_train_counties': len(config['dataset']['train_counties']),
        'housing': config['model']['args']['use_housing'],
        'sigma': config['dataset']['kwargs'].get('sigma', 3)
    }

    dwargs = config['dataset']['kwargs']
    if dwargs['use_fill_frac']:
        ret['objective'] = 'Fill fraction'
    elif dwargs['use_max_scaling']:
        ret['objective'] = 'Relative density'
    else:
        ret['objective'] = 'Absolute density'
    return ret