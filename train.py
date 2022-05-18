import os
import time
import sys
import copy
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error

from model import LatentBinModel
from dataset import load_data, MetaDataset, FewShotCollator

device = "cuda" if t.cuda.is_available() else "cpu"


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model: nn.Module, optimizer: t.optim,
          train_loader: DataLoader, valid_loader: DataLoader, args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    best_loss = float('inf')
    train_metrics = {}
    val_metrics = {}
    train_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        truth_list, pred_list = [], []
        train_loss_list = []
        for data in tqdm(train_loader):
            adjust_learning_rate(optimizer, train_step+1)
            context_x, context_y, target_x, target_y, target_y_f = data
            context_x = context_x.to(device)
            context_y = context_y.to(device)
            target_x = target_x.to(device)
            target_y = target_y.to(device)
            target_y_f = target_y_f.to(device)

            y_pred, sigma, kl, loss = model(context_x, context_y, target_x, target_y, target_y_f)
            print(kl, loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1

            # get target items
            target_y_r = target_y_f[:, context_y.size()[-2]:, :]
            y_pred_r = y_pred[:, context_y.size()[-2]:, :]

            # get non-zero item index
            nonzero_idx = t.where(target_y_r > 0, True, False)
            nonzero_target_y = target_y_r[nonzero_idx]
            nonzero_y_pred = y_pred_r[nonzero_idx]

            truth_list += nonzero_target_y.squeeze().detach().cpu().tolist()
            pred_list += nonzero_y_pred.squeeze().detach().cpu().tolist()

            train_loss_list += [loss.item()]
        truth_list = np.array(truth_list).squeeze()
        pred_list = np.array(pred_list).squeeze()

        train_metrics['mae'] = mean_absolute_error(truth_list, pred_list)
        train_metrics['loss'] = sum(train_loss_list)/len(train_loss_list)

        model.eval()
        with t.no_grad():
            truth_list, pred_list = [], []
            valid_loss_list = []
            for data in tqdm(valid_loader):
                context_x, context_y, target_x, target_y, target_y_f = data

                context_x = context_x.to(device)
                context_y = context_y.to(device)
                target_x = target_x.to(device)
                target_y = target_y.to(device)
                target_y_f = target_y_f.to(device)

                y_pred, sigma, kl, loss = model(context_x, context_y, target_x, target_y, target_y_f)

                target_y_r = target_y_f[:, context_y.size()[-2]:, :]
                y_pred_r = y_pred[:, context_y.size()[-2]:, :]

                nonzero_idx = t.where(target_y_r > 0, True, False)
                nonzero_target_y = target_y_r[nonzero_idx]
                nonzero_y_pred = y_pred_r[nonzero_idx]

                truth_list += nonzero_target_y.squeeze().detach().cpu().tolist()
                pred_list += nonzero_y_pred.squeeze().detach().cpu().tolist()

                valid_loss_list += [loss.item()]
            truth_list = np.array(truth_list).squeeze()
            pred_list = np.array(pred_list).squeeze()

            val_metrics['mae'] = mean_absolute_error(truth_list, pred_list)
            val_metrics['loss'] = sum(valid_loss_list)/len(valid_loss_list)

        print(f'Epoch[{epoch}/{epochs}] |',
              f'train_loss:{train_metrics["loss"]:.3f} |',
              f'train_mae:{train_metrics["mae"]:.3f} |',
              f'val_loss:{val_metrics["loss"]:.3f} |',
              f'val_mae:{val_metrics["mae"]:.3f} |')

        # Save best loss model
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            best_model = copy.deepcopy(model)
            trigger_count = 0
            t.save({
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
                 }, os.path.join(args.logdir, "model.pt"))

    return best_model


if __name__ == '__main__':
    # Model arguments
    parser = ArgumentParser(description='MetaDTA argument')
    parser.add_argument('--name', default = 'default', dest='name', type=str, help='Model Name')
    parser.add_argument('--d_model', default=128, type=int, help='Hidden Space dimension')
    parser.add_argument('--n_CA', default=2, type=int, help='Layer Number of MultiHead CrossAttention')
    parser.add_argument('--n_SA', default=2, type=int, help='Layer Number of MultiHead SelfAttention')
    parser.add_argument('--input', default='Bin', type=str, help='Data Input Type. [Scalar, Bin]')
    parser.add_argument('--use_latent_path', default=False, action='store_true',
                        help="")

    # Data arguments
    parser.add_argument('--n_bins', default=32,
                        help='Number of histogram bins for binding affinity')
    parser.add_argument('--seq_len', default=512, type=int,
                        help='Ligand Count per Target. Sequence Length')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of batch size")

    # Training argument
    parser.add_argument('--lr', type=float, default='1e-4', help="")
    parser.add_argument('--n_epochs', type=int, default=1000, help="")

    # Utility argument
    args = parser.parse_args()

    args.logdir = "./runs/"

    n_bins = args.n_bins
    n_CA = args.n_CA
    n_SA = args.n_SA
    seq_len = args.seq_len
    ligand_cnt = 100 ## number of support set
    use_latent_path = args.use_latent_path

    train_coo, test_coo, total_ecfp = load_data()

    input_dim = total_ecfp.shape[-1]
    train_list = list(set(train_coo.col))   # MetaDTA Training mode
    test_list = list(set(test_coo.col))

    train_set = MetaDataset(train_list, total_ecfp, train_coo, n_bins, seq_len=seq_len)
    test_set = MetaDataset(test_list, total_ecfp, test_coo, n_bins, seq_len=seq_len)

    traincollator = FewShotCollator()
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              collate_fn=traincollator, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    testcollator = FewShotCollator(ligand_cnt=ligand_cnt)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                            collate_fn=testcollator, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    epochs = args.n_epochs
    d_model = args.d_model

    model = LatentBinModel(input_dim, n_bins, d_model, n_CA, n_SA, use_latent_path).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=args.lr)

    best_model = train(model, optimizer, train_loader, test_loader, args)

