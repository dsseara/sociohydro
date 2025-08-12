import os
import glob
import torch
import torch.nn.functional as F
import json
import yaml
import numpy as np

from argparse import ArgumentParser
from time import time

from census_dataset import CensusDataset
import census_nn
import census_unet

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_training(model, 
                 train_dataset, 
                 val_dataset, 
                 optimizer, 
                 scheduler,
                 save_dir, 
                 n_epochs=100,
                 batch_size=8):
    """ Train a model """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model.to(device)

    step = 0
    best_loss = 1e10
    best_epoch = 0

    idxs = np.arange(len(train_dataset), dtype=int)
    logger.info('Starting to train')
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        np.random.shuffle(idxs)
        t = time()

        # Training loop
        for i in range(len(train_dataset)):
            batch = train_dataset[idxs[i]]
            wb = batch['wb'].to(device) #Shape [2, 2, H, W]
            housing = batch['housing'][None, None].to(device) # Some models use housing
            mask = batch['mask'] # Only use points in county for loss

            wb[wb.isnan()] = 0.
            housing[housing.isnan()] = 0.

            wbNN = wb[0:1] + batch['dt'] * model(wb[0:1], housing=housing)
            loss = F.l1_loss(wbNN[:, :,mask], wb[1:2,:,mask])

            loss.backward()
            train_loss += loss.item()
            step += 1

            if step % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        train_loss = train_loss / len(train_dataset)

        val_loss = 0.
        val_err = 0.
        model.eval()

        # Our evaluation metric is the forecasting error over the full county time series
        with torch.no_grad():
            for dataset in val_dataset.datasets:
                batch = dataset[0]
                wb = batch['wb'].to(device)
                housing = batch['housing'][None, None].to(device) # Some models use housing

                wb[wb.isnan()] = 0.
                housing[housing.isnan()] = 0.

                mask = batch['mask'] # Only use points in county for loss

                wbNN = model.simulate(wb[0:1], n_steps=wb.shape[0]-1, dt=batch['dt'], housing=housing)[0]

                diff = wbNN[-1,:,mask] - wb[-1,:,mask]
                housing = housing[0,:,mask]

                val_loss += diff.pow(2).sum(0).mean().item()

                if dataset.use_fill_frac:
                    val_err += (diff * housing).pow(2).sum(0).mean().item()
                elif dataset.use_max_scaling:
                    val_err += (diff * housing.max()).pow(2).sum(0).mean().item()
                else:
                    val_err += diff.pow(2).sum(0).mean().item()

        scheduler.step()
        logger.info(f'Epoch {epoch}\tTrain Loss = {train_loss:.3g}\tVal Loss = {val_loss:.3g}\tVal Err = {val_err:.3g}\t{time()-t:.3g} s')

        # Save if the model showed an improvement
        if val_err <= best_loss:
            save_dict = {
                'state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_err': val_err,
            }
            torch.save(save_dict, f'{save_dir}/model_weight.ckpt')
            best_loss = val_err
            best_epoch = epoch
        
        # Early stopping if the model is no longer improving
        if epoch - best_epoch > 20:
            return


def get_dataset(config, random_seed=42):
    train_datasets = [CensusDataset(county, **config['kwargs']).training() for county in config['train_counties']]
    val_datasets = [CensusDataset(county, **config['kwargs']).validate() for county in config['val_counties']]
    
    # Include first decade of Census data from Figure 3 counties
    for county in config['val_counties']:
        train_datasets.append(torch.utils.data.Subset(
            CensusDataset(county, **config['kwargs']).training(), 
            np.arange(0, config['val_tmax'], dtype=int)
        ))

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    logger.info(f'Training dataset length: {len(train_dataset)}')
    logger.info(f'Validation dataset length: {len(val_dataset)}')

    return train_dataset, val_dataset

def get_model(config):
    class_type = eval(config['class_path'])
    logger.info(f'Building a {class_type.__name__}')
    model = class_type(**config['args'])
    
    if 'weights' in config and config['weights'] is not None:
        logger.info(f'Loading model weights from {config["weights"]}')
        info = torch.load(config['weights'], map_location='cpu')
        model.load_state_dict(info['state_dict'])

        logger.info(f'Model reached error={info["val_err"]:.3g}')
    
    return model

def get_optimizer_scheduler(config, model):
    class_type = eval(config['optimizer']['class_path'])
    logger.info(f'Building a {class_type.__name__}')
    optimizer = class_type(model.parameters(), **config['optimizer']['args'])
    if 'scheduler' in config:
        class_type = eval(config['scheduler']['class_path'])
        logger.info(f'Building a {class_type.__name__}')
        scheduler = class_type(optimizer, **config['scheduler']['args'])
    else:
        logger.info('Proceeding with no learning rate scheduler')
        scheduler = None
    
    return optimizer, scheduler


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/absolute_density_no_housing.yaml')
    parser.add_argument('--model_id', type=str, default='test')
    args = parser.parse_args()

    logger.info(f'Loading configuration from {args.config}')
    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Load model
    model = get_model(config['model'])

    # Load datasets
    train_dataset, val_dataset = get_dataset(config['dataset'])

    # Load optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(config, model)

    # Dump configuration with model weihgt save path
    save_dir = f'models/{args.model_id}'
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f'Saving information to {save_dir}')
    config['model']['weights'] = f'{save_dir}/model_weight.ckpt'
    with open(f'{save_dir}/config.yaml', 'w') as file:
        yaml.dump(config, file)

    with torch.autograd.set_detect_anomaly(True):
        run_training(
            model, 
            train_dataset, 
            val_dataset, 
            optimizer,
            scheduler,
            save_dir=save_dir,
            **config['training'])
