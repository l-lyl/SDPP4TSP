import json
from functools import partial
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import pickle as cPickle

class TemporalSetsDataset(Dataset):

    def __init__(self, data_path, data_info_path, data_type='train'):
        with open(data_path, 'r') as file:
            data_dict = json.load(file)
        self.max_num_sets = 3  #2
        
        self.sets = []
        self.times = []
        self.users = []
        self.targets = []
        
        for user in data_dict:
            sets = user['sets']
            num_sets = len(sets)
            if data_type == 'train':
                for idx in range(1, num_sets - 2): 
                    self._add_data(sets, user['user_id'], idx, data_type)
            elif data_type == 'validate':
                self._add_data(sets, user['user_id'], num_sets - 2, data_type) 
            elif data_type == 'test':
                self._add_data(sets, user['user_id'], num_sets - 1, data_type)
            else:
                raise NotImplementedError()
        assert len(self.sets) == len(self.targets)
        assert len(self.sets) == len(self.times)
        
        with open(data_info_path, 'r') as file:
            data_info_dict = json.load(file)
        self.items_total = data_info_dict['num_items']
        
        arrange_sets = {}
        arrange_times = {}
        arrange_users = {}
        arrange_targets = {}
        index = 0
        for sets in self.sets:
            set_num = len(sets) + len(self.targets[index])
            if set_num not in arrange_sets.keys():
                arrange_sets[set_num] = [sets]
                arrange_times[set_num] = [self.times[index]]
                arrange_users[set_num] = [self.users[index]]
                arrange_targets[set_num] = [self.targets[index]]
            else:
                arrange_sets[set_num].append(sets)
                arrange_times[set_num].append(self.times[index])
                arrange_users[set_num].append(self.users[index])
                arrange_targets[set_num].append(self.targets[index])
            index += 1
        lsets = []
        ltimes = []
        lusers = []
        ltargets = []
        for k, v in arrange_sets.items():
            lsets += v
        for k, v in arrange_times.items():
            ltimes += v
        for k, v in arrange_users.items():
            lusers += v
        for k, v in arrange_targets.items():
            ltargets += v
            
        self.sets = lsets
        self.times = ltimes
        self.users = lusers
        self.targets = ltargets

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        return self.sets[idx], self.times[idx], self.users[idx], self.targets[idx]

    def _add_data(self, sets, user_id, idx, datatype):
        user_sets, times = [], []
        select_sets = sets[:idx] if idx - self.max_num_sets < 0 else sets[idx - self.max_num_sets : idx]
        for user_set in select_sets:
            user_sets.append(user_set['items'])
            # times.append(user_set['timestamp'])
            times.append(sets[idx]['timestamp'] - user_set['timestamp'])
        self.sets.append(user_sets)
        self.times.append(times)
        self.users.append(user_id)
        
        target_sets = []
        if datatype == "train":
            pre_len, all_len = len(user_sets), len(sets) 
            for iid in range(idx, all_len-2):
                target_sets.append(sets[iid]['items'])
                if len(target_sets) == pre_len:
                    break
            self.targets.append(target_sets)
        else:
            self.targets.append(sets[idx]['items']) 
        
class TemporalSetsInput(object):

    def __init__(self, sets_batch, times_batch, users_batch, targets_batch, items_total, k_kernel, model_mode):
        """
        Args:
            sets_batch (List[List[List]]): shape (batch_size, num_sets, num_items)
            times_batch (List[List]): shape (batch_size, num_sets)
        """
        self.items_total = items_total
        self.sets_batch = [[torch.tensor(user_set) for user_set in user_sets] for user_sets in sets_batch]
        self.times_batch = [torch.tensor(user_times) for user_times in times_batch]
        ##new add
        self.model_mode = model_mode
        if self.model_mode == 'train':
            self.targets_batch = [[torch.tensor(target) for target in targets] for targets in targets_batch]
            
            negatives_batch = self.generate_negatives(targets_batch, sets_batch)
            self.negatives_batch =  [[torch.tensor(negative) for negative in negatives] for negatives in negatives_batch]
            
            k_kernels_batch = self.generate_k_kernels(k_kernel)
            self.k_kernels_batch = k_kernels_batch
        else:
            self.targets_batch = targets_batch
        
        self.users_batch = torch.tensor(users_batch)
        self.batch_size = len(sets_batch)

    def get_users(self) -> torch.Tensor:
        return self.users_batch

    def get_items(self) -> List[torch.Tensor]:
        """
        Returns:
            output (List[Tensor]): shape (batch_size, num_items)
        """
        items_batch = []
        for user_sets in self.sets_batch:
            user_items = torch.cat([user_set for user_set in user_sets], dim=-1)
            user_items = user_items[-512:] if len(user_items) > 512 else user_items
            items_batch.append(user_items)
        return items_batch

    def get_item_times(self) -> List[torch.Tensor]:
        """
        Returns:
            output (List[Tensor]): shape (batch_size, num_items)
        """
        times_batch = []
        for user_sets, user_times in zip(self.sets_batch, self.times_batch):
            user_items_times = []
            for user_set, time in zip(user_sets, user_times):
                user_items_times.append(user_set.new_full(user_set.shape, time))
            user_items_times = torch.cat(user_items_times, dim=-1)
            user_items_times = user_items_times[-512:] if len(user_items_times) > 512 else user_items_times
            times_batch.append(user_items_times)
        return times_batch
    
    def get_sets(self) -> List[List[torch.Tensor]]:
        """
        Returns:
            output: shape (batch_size, num_sets, num_items)
        """
        return self.sets_batch

    def get_set_times(self) -> List[torch.Tensor]:
        """
        Returns:
            output: shape (batch_size, num_sets)
        """
        return self.times_batch
    
    def get_set_targets(self) -> List[torch.Tensor]: 
        return self.targets_batch
    
    def get_set_negatives(self) -> List[torch.Tensor]: 
        return self.negatives_batch

    def get_k_kernels(self) -> List[torch.Tensor]: 
        return self.k_kernels_batch

    def cuda(self, device):
        self.sets_batch = [[user_set.cuda(device) for user_set in user_sets] for user_sets in self.sets_batch]
        self.times_batch = [user_times.cuda(device) for user_times in self.times_batch]
        self.users_batch = self.users_batch.cuda(device)
        if self.model_mode == 'train':
            self.negatives_batch = [[negative.cuda(device) for negative in negatives] for negatives in self.negatives_batch]
            self.k_kernels_batch = [kernels.cuda(device) for kernels in self.k_kernels_batch]
            self.targets_batch = [[target.cuda(device) for target in targets] for targets in self.targets_batch]
        return self

    def generate_negatives(self, positives_batch, sets_batch):
        negatives_batch = []
        pre_tar_sets = []
        for i in range(len(sets_batch)):
           pre_tar_sets.append(sets_batch[i] + positives_batch[i])
        for pos_sets in pre_tar_sets:
            positives = []
            for pos_set in pos_sets:
                for pos_i in pos_set:
                    positives.append(pos_i)
            neg_sets = []
            for pos_set in pos_sets:
                neg_set = []
                for _ in range(len(pos_set)):
                    negative = positives[0]
                    while negative in positives or negative in neg_set:
                        negative = np.random.randint(0, self.items_total)
                    neg_set.append(negative)
                neg_sets.append(neg_set)
            negatives_batch.append(neg_sets)
        return negatives_batch
        
    def generate_k_kernels(self, k_kernel):
        kernels_batch = []
        index = 0
        for sets in self.sets_batch:
            new_sets = [aset for aset in sets] 
            for tset in self.targets_batch[index]: 
                new_sets.append(tset)   
            for nset in self.negatives_batch[index]:
                new_sets.append(nset) 

            ground_items = torch.cat(new_sets, 0)
            sub_k_kernel = k_kernel[ground_items][:, ground_items]
            len_list = [len(a_set) for a_set in new_sets]
            set_num = len(new_sets)
            set_kernel = torch.zeros(set_num, set_num)
           
            row_s, col_s = 0, 0
            row_e, col_e = 0, 0
            for i in range(set_num):
                row_e += len_list[i]
                col_e, col_s = 0, 0
                for j in range(set_num):
                    col_e += len_list[j]
                    set_kernel[i][j] = sub_k_kernel[row_s:row_e, col_s:col_e].mean()
                    col_s = col_e
                row_s = row_e
            kernels_batch.append(torch.sigmoid(set_kernel)) 
            index += 1
        return kernels_batch

def collate_fn(data, items_total, k_kernel=None, model_mode="train", negative_sample=False, num_samples=10):
    
    sets_batch = [sets for sets, times, user, targets in data]
    times_batch = [times for sets, times, user, targets in data]
    users_batch = [user for sets, times, user, targets in data]
    
    if model_mode == 'train':
        targets_batch = [targets for sets, times, user, targets in data] 
    else:
        targets_batch = torch.stack([torch.zeros(items_total).index_fill_(0, torch.tensor(target), 1) for _, _, _, target in data])

    temporal_sets_input = TemporalSetsInput(sets_batch=sets_batch,
                                            times_batch=times_batch,
                                            users_batch=users_batch,
                                            targets_batch=targets_batch,
                                            items_total=items_total,
                                            k_kernel=k_kernel,
                                            model_mode=model_mode)

    return temporal_sets_input

def get_temporal_sets_data_loader(data_path, data_info_path, batch_size, pre_kernel_path, negative_sample=False, num_samples=10):
    
    train_dataset = TemporalSetsDataset(data_path, data_info_path, data_type='train')
    validate_dataset = TemporalSetsDataset(data_path, data_info_path, data_type='validate')
    test_dataset = TemporalSetsDataset(data_path, data_info_path, data_type='test')
    
    lk_param = cPickle.load(open(pre_kernel_path, 'rb'), encoding="latin1")
    lk_tensor = torch.FloatTensor(lk_param['V'])
    lk_tensor = F.normalize(lk_tensor, p=2, dim=1)
    k_kernel = torch.matmul(lk_tensor, lk_tensor.t())
    
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   collate_fn=partial(collate_fn, k_kernel=k_kernel, model_mode="train", items_total=train_dataset.items_total, negative_sample=negative_sample, num_samples=num_samples))
    
    validate_data_loader = DataLoader(dataset=validate_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=partial(collate_fn, model_mode="evaluate", items_total=validate_dataset.items_total))
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=partial(collate_fn,  model_mode="test", items_total=test_dataset.items_total))
    return train_data_loader, validate_data_loader, test_data_loader


if __name__ == '__main__':
    train_data_loader, validate_data_loader, test_data_loader = get_temporal_sets_data_loader(data_path='data/taobao_buy-1.json',
                                                                                              data_info_path='data/taobao_buy-1_info.json',
                                                                                              pre_kernel_path='data/taobao_buy-1_kernel.pkl',
                                                                                              batch_size=4)
    '''
    for idx, inputs in enumerate(train_data_loader):
        print("+++++++sets_batch\n", inputs.get_sets())
        print("*******user batch\n", inputs.get_users())
        print(".......targets\n", inputs.get_set_targets())
        print("-------negative batch\n", inputs.get_set_negatives()) 
        print(a)
    '''