from unicodedata import category
import torch
from PIL import Image
import os
import os.path as osp
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader 
import json


class CustomDataset(Dataset):
    def __init__ (self,json_files = [], transform=None, igore_abnormal=True):
        super(CustomDataset,self).__init__()
        
        self.transform = transform
        self.categories = np.arange(24)
        self.images = []
        self.labels = []
        self.label2image_idx = {}
        start = 0
        if not isinstance(json_files, list):
            json_files = [json_files]

        for json_f in json_files:
            assert osp.exists(json_f), 'no json file found:'+json_f
            f_root = osp.dirname(json_f)
            with open(json_f, 'r') as f:
                data = json.load(f)
                for patient in data['items']:
                    if igore_abnormal and patient['kar_type']!='normal':
                        continue
                    for kar_file in patient['kar_files']:
                        for item in patient[kar_file]:
                            patch_file, label = item
                            if not osp.exists(osp.join(f_root, patch_file)):continue
                            label = label-1
                            self.images.append(osp.join(f_root, patch_file))
                            self.labels.append(label)
                            if label not in self.label2image_idx:
                                self.label2image_idx[label] = []

                            self.label2image_idx[label].append(start)
                            start+=1

        print('total samples:', len(self))
      
    
    def __len__(self):
        total = len(self.images)
        return total
    
    def __getitem__(self, index):
        im_path = self.images[index]
        label = self.labels[index]
        image=Image.open(im_path).convert('RGB')
        if self.transform is not None:
            image= self.transform(image)
        image = np.asarray(image)/255.0
      
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label)
        return image, label

class CustomBatchSampler(Sampler):
    def __init__(self,dataset, batch_size_per_cls=1):
        super(CustomBatchSampler,self).__init__(dataset)
        self.dataset = dataset
        self.batch_size_per_cls = batch_size_per_cls
        self.n_categories = len(self.dataset.categories)
        self.offset_per_cls = [0 for i in range(self.n_categories)]

    def __iter__(self):
        batch =[]
        i = 0
        while i<len(self):  
            for c in self.dataset.categories:
                for k in range(self.batch_size_per_cls):

                    c_offset = (self.offset_per_cls[c]+k)%len(self.dataset.label2image_idx[c])
                    #print(len(self.dataset.label2image_idx[c]), c, k, c_offset)
                    c_im_index = self.dataset.label2image_idx[c][c_offset]
                    batch.append(c_im_index)
                self.offset_per_cls[c]+=self.batch_size_per_cls
                    
                if self.offset_per_cls[c]>len(self.dataset.label2image_idx[c]):
                    self.offset_per_cls[c] =0
                    np.random.shuffle(self.dataset.label2image_idx[c])

            assert len(batch) == (self.n_categories*self.batch_size_per_cls)
            
            yield batch
            batch = []
            i+=1
           
    def __len__(self):
        return len(self.dataset)//(len(self.dataset.categories)*self.batch_size_per_cls)
