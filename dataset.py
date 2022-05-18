import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def get_score_binning(mu_arr, sigma, n_bins):
    # Scalar to Bin Distribution
    # Input : Scalr (N,)
    # Output : n_bins Bin Class
    #          (Normal Distribution between 0 to 11) (N, n_bins)
    nonzero_mu_arr = mu_arr[mu_arr != 0]
    n_pad = sum(mu_arr == 0)
    samples = [
        np.random.normal(mu, sigma*0.05, 2000) if
        mu > 0 else np.array([0]*2000) for mu in nonzero_mu_arr]
    count = [
        np.histogram(
            sample,
            bins=np.linspace(0, 11, n_bins+1))[0]/2000 for sample in samples]
    count.extend([[0]*n_bins]*n_pad)
    count = np.array(count)
    return count


def load_data():
    DATA_PATH = './data'
    with open(os.path.join(DATA_PATH, 'train_coo.pkl'), 'rb') as f:
        train_coo = pickle.load(f)
    with open(os.path.join(DATA_PATH, 'test_coo.pkl'), 'rb') as f:
        test_coo = pickle.load(f)

    ecfp = np.load(
        os.path.join(DATA_PATH, "total_ecfp.npy"), allow_pickle=True)
    return train_coo, test_coo, ecfp


class MetaDataset(Dataset):
    # Train/Val/Test are sepearated
    def __init__(self, target_list, total_ecfp, coo, n_bins=32, seq_len=512):
        self.target_list = target_list
        self.total_ecfp = total_ecfp
        self.coo_data, self.coo_row, self.coo_col = coo.data, coo.row, coo.col
        self.n_bins = n_bins
        self.seq_len = seq_len

    def __getitem__(self, idx):
        target_num = self.target_list[idx]
        ligand_num, ligand_data, ligand_data_bin, ecfp = {}, {}, {}, {}

        train_index = np.where(self.coo_col == target_num)
        ligand_num = self.coo_row[train_index]
        ligand_data = self.coo_data[train_index]
        ligand_data_bin = get_score_binning(ligand_data, np.std(ligand_data), self.n_bins)
        ecfp = np.array([self.total_ecfp[num] for num in ligand_num])

        return {'target': target_num,
                'ecfp': ecfp,
                'ligand_data': ligand_data,
                'ligand_data_bin': ligand_data_bin,
                'ligand_num': ligand_num,
                'seq_len': self.seq_len,
                'n_bins': self.n_bins
                }

    def __len__(self):
        return len(self.target_list)


class FewShotCollator():
    def __init__(self, ligand_cnt=None):
        self.ligand_cnt = ligand_cnt

    def __call__(self, batch):
        max_ligand_num = batch[0]['seq_len']

        if self.ligand_cnt is None:
            num_context = np.random.randint(10, round(max_ligand_num*0.85))  # for training
        else:
            num_context = self.ligand_cnt   # for few-shot evaluation

        context_x, context_y, target_x, target_y, target_y_f = list(), list(), list(), list(), list()
        context_y_f =  list()
        # Sampling context ligand
        # If len(ligand_num) >= num_context * 0.85 , use all context.
        # Else, pad dummy data
        for data in batch:
            ligand_data = data['ligand_data']
            ligand_data_bin = data['ligand_data_bin']
            ligand_num = data['ligand_num']
            ecfp = data['ecfp']
            total_idx = np.random.choice(range(len(ligand_num)), min(max_ligand_num, len(ligand_num)), replace=False)

            if len(ligand_num) < 10:
                continue
            if len(ligand_num) * 0.85 >= num_context:
                c_idx = total_idx[:num_context]
            else:
                c_idx = total_idx[:round(len(ligand_num)*0.85)]

            c_y = torch.FloatTensor(ligand_data_bin[c_idx])
            c_x = torch.FloatTensor(ecfp[c_idx])
            c_y_f = torch.FloatTensor(ligand_data[c_idx])

            t_idx = total_idx[len(c_idx):]

            t_y = torch.FloatTensor(ligand_data_bin[t_idx])
            t_y_f = torch.FloatTensor(ligand_data[t_idx])
            t_x = torch.FloatTensor(ecfp[t_idx])

            context_x.append(c_x)
            context_y.append(c_y)
            context_y_f.append(c_y_f)
            target_x.append(t_x)
            target_y.append(t_y)
            target_y_f.append(t_y_f)

        context_x = pad_sequence(context_x, batch_first=True, padding_value=0)
        context_y = pad_sequence(context_y, batch_first=True, padding_value=0)
        context_y_f = pad_sequence(context_y_f, batch_first=True, padding_value=0).unsqueeze(-1)

        target_x = pad_sequence(target_x, batch_first=True, padding_value=0)
        target_y = pad_sequence(target_y, batch_first=True, padding_value=0)
        target_y_f = pad_sequence(target_y_f, batch_first=True, padding_value=0).unsqueeze(-1)

        target_x = torch.cat([context_x, target_x], dim=1)
        target_y = torch.cat([context_y, target_y], dim=1)
        target_y_f = torch.cat([context_y_f, target_y_f], dim=1)

        return context_x, context_y, target_x, target_y, target_y_f
