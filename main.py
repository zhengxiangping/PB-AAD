import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import random
import swanlab
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from basicts.utils import load_adj, load_pkl
from data import PretrainingDataset, ForecastingDataset
from basicts.data import SCALER_REGISTRY
from basicts.losses import sce_loss
from basicts.mask.model import pretrain_model, finetune_model
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.utils.utils import metric_forward, select_input_features

metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(args):
    return torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
def get_dataloader(dataset_class, dataset_dir, dataset_index_dir, mode, seq_len, batch_size, num_workers=8, shuffle=False):
    dataset = dataset_class(dataset_dir, dataset_index_dir, mode, seq_len)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def evaluate(data_loader, model, config, scaler, mode="val"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in tqdm(data_loader, ncols=100):
            future_data, history_data, long_history_data = data
            history_data = select_input_features(history_data, config['target_features']).to(config['device'])
            long_history_data = select_input_features(long_history_data, config['forward_features']).to(config['device'])
            future_data = select_input_features(future_data, config['target_features']).to(config['device'])
            pred = model(history_data, long_history_data, future_data, future_data.shape[0], 0)
            preds.append(pred.detach().cpu())
            labels.append(future_data.detach().cpu())
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = SCALER_REGISTRY.get(scaler["func"])(preds, **scaler["args"])
    labels = SCALER_REGISTRY.get(scaler["func"])(labels, **scaler["args"])
    results = {name: metric_forward(func, [preds, labels]).item() for name, func in metrics.items()}
    print(f"Evaluate {mode} data: " + ", ".join([f"{k}: {v:.4f}" for k, v in results.items()]))
    return results

def train_epoch(data_loader, model, optimizer, scaler, config):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, ncols=100):
        future_data, history_data, long_history_data = data
        history_data = select_input_features(history_data, config['target_features']).to(config['device'])
        long_history_data = select_input_features(long_history_data, config['forward_features']).to(config['device'])
        future_data = select_input_features(future_data, config['target_features']).to(config['device'])
        preds = model(history_data, long_history_data, future_data, future_data.shape[0], 0)
        preds_rescaled = SCALER_REGISTRY.get(scaler["func"])(preds, **scaler["args"])
        labels_rescaled = SCALER_REGISTRY.get(scaler["func"])(future_data, **scaler["args"])
        loss = metric_forward(masked_mae, [preds_rescaled, labels_rescaled])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f"Train loss: {avg_loss:.4f}")
    return avg_loss

def finetune(config, args):
    print('### start finetune ... ###')
    config['device'] = get_device(args)
    scaler = load_pkl(config['scaler_dir'])
    adj_mx, _ = load_adj(config['adj_dir'], "doubletransition")
    config['backend_args']['supports'] = [torch.tensor(i) for i in adj_mx]
    train_loader = get_dataloader(
        ForecastingDataset,
        config['dataset_dir'],
        config['dataset_index_dir'],
        'train',
        config['seq_len'],
        config['batch_size'],
        shuffle=True
    )
    val_loader = get_dataloader(
        ForecastingDataset,
        config['dataset_dir'],
        config['dataset_index_dir'],
        'valid',
        config['seq_len'],
        config['batch_size']
    )
    test_loader = get_dataloader(
        ForecastingDataset,
        config['dataset_dir'],
        config['dataset_index_dir'],
        'test',
        config['seq_len'],
        config['batch_size']
        
    )
    model = finetune_model(config['pre_trained_path'], config['mask_args'], config['backend_args'], config.get('backbone', 'gwnet')).to(config['device'])
    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=1e-5, eps=1e-8)
    for epoch in range(config['finetune_epochs']):
        print(f'============ epoch {epoch} ============')
        train_epoch(train_loader, model, optimizer, scaler, config)
        print('============ val and test ============')
        val_results = evaluate(val_loader, model, config, scaler, mode="val")
        test_results = evaluate(test_loader, model, config, scaler, mode="test")
        swanlab.log({f"val_{k}": v for k, v in val_results.items()}, step=epoch)
        swanlab.log({f"test_{k}": v for k, v in test_results.items()}, step=epoch)
    return val_results, test_results
def pretrain(config, args):
    print('### start pre-training ... ###')
    config['device'] = get_device(args)
    scaler = load_pkl(config['preTrain_scaler_dir'])
    train_loader = get_dataloader(
        PretrainingDataset,
        config['preTrain_dataset_dir'],
        config['preTrain_dataset_index_dir'],
        'train',
        config['seq_len'],
        config['preTrain_batch_size'],
        num_workers=16,
        shuffle=True
    )
    model = pretrain_model(
        config['num_nodes'], config['dim'], config['topK'], config['adaptive'],
        config['pretrain_epochs'], config['patch_size'], config['in_channel'],
        config['embed_dim'], config['num_heads'], config['mlp_ratio'],
        config['dropout'], config['mask_ratio'], config['encoder_depth'], config['decoder_depth']
    )
    device_ids = list(range(torch.cuda.device_count()))
    if device_ids:
        model = DataParallel(model, device_ids=device_ids).to(device_ids[0])
    else:
        model = model.to(config['device'])
    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=1e-5, eps=1e-8)
    lossType = masked_mae if args.lossType == 'mae' else sce_loss
    for epoch in range(config['pretrain_epochs']):
        print(f'============ epoch {epoch} ============')
        model.train()
        total_loss = 0
        for idx, data in enumerate(tqdm(train_loader, ncols=100)):
            _, history_data = data
            history_data = select_input_features(history_data, config['forward_features']).to(config['device'])
            
            outputs = model(history_data, epoch)
            reconstruction_masked_tokens = outputs['reconstruction']
            label_masked_tokens = outputs['labels']
            sparsity_loss = outputs['sparsity_loss']
            main_loss = metric_forward(lossType, [reconstruction_masked_tokens, label_masked_tokens])
            loss = main_loss + sparsity_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"preTrain loss: {avg_loss:.4f}")
        swanlab.log({"pretrain_loss": avg_loss, "epoch": epoch})
        
        print("Saving Model ...")
        pre_trained_path = config["model_save_path"] + "checkpoint_" + str(epoch) +".pt"
        if isinstance(model, DataParallel):
            torch.save(model.module.state_dict(), pre_trained_path)
        else:
            torch.save(model.state_dict(), pre_trained_path)
        print(f"Model saved to {pre_trained_path}")
        config['pre_trained_path'] = pre_trained_path
        # config['finetune_epochs'] = 1
        # val_results, test_results =finetune(config, args)
        # swanlab.log({f"pre_val_{k}": v for k, v in val_results.items()}, step=epoch)
        # swanlab.log({f"pre_test_{k}": v for k, v in test_results.items()}, step=epoch)
    return model

def main(config, args):
    set_seed(0)
    if args.preTrain == 'true':
        model = pretrain(config, args)
    else:
        finetune(config, args)

def update_config(config, args):
    device = get_device(args)
    config['device'] = device
    config['preTrain_batch_size'] = args.preTrain_batch_size
    config['batch_size'] = args.batch_size
    config['pretrain_epochs'] = args.pretrain_epochs
    config['finetune_epochs'] = args.finetune_epochs
    config['mask_ratio'] = args.mask_ratio
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='./parameters/PEMS03.yaml', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--preTrain', default='true', type=str)
    parser.add_argument('--lossType', default='mae', type=str)
    parser.add_argument('--preTrain_batch_size', default=8, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--finetune_epochs', default=100, type=int)
    parser.add_argument('--mask_ratio', default=0.25, type=float)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = update_config(config, args)
    
    run = swanlab.init(
        project="GPT-GNN",  # 项目名称
        config=config
    )
    main(config, args)




