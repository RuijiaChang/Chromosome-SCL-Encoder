from models.cae import CAE
import numpy as np
import torch
from datasets.data_loader import CustomDataset, CustomBatchSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from cfgs.config import cfg
import torch.nn.functional as F
from trainers.cae_trainer import Trainer
import argparse
import glob
import os.path as osp
from PIL import Image
import warnings

def get_args():
    parser = argparse.ArgumentParser(description='Train the CAE for chromosome anormaly detection')
    parser.add_argument('--encoder', '-encoder', type=str, default='resnet50', help='The backbone for feature extraction,\
                        optional: resnet34, resnet50, resnet101, mit_b0, mit_b1, mit_b2,...')
    parser.add_argument('--decoder', '-decoder', type=str, default='unet', help='The structure for feature restoring,\
                        optional: unet, fpn,...')
    parser.add_argument('--input_size', '-input_size', type=int, default=224, help='the feed size of image')
    parser.add_argument('--data_aug', '-data_aug', type=str, default='', help='the augmentation for dataset')
    parser.add_argument('--load', '-load', type=str, default='weights_best.h5', help='Load model from a .h5 file')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='your save path', help='the path to save weights')
    parser.add_argument('--epochs', '-epochs', type=int, default=120, help='Epochs for training')
    parser.add_argument('--steps_per_epoch', '-steps_per_epoch', type=int, default=0, help='iterations for each epoch')
    parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='Base learning rate for training')
   
    return parser.parse_args()
    

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = get_args()
    
    #define the transforms for data augmentation
    data_transforms = []   
    for aug in args.data_aug.split('-'):
        if 'rot' in aug:
            data_transforms.append(transforms.RandomRotation(degrees=90, expand=False, fill=255))
        if 'jit' in aug:
            data_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        if 'crop' in aug:
            data_transforms.append(transforms.RandomResizedCrop(cfg.INPUT_SHAPE, scale=(0.9, 1.0), ratio=(0.9, 0.9)))
        if 'hflip' in aug:
            data_transforms.append(transforms.RandomHorizontalFlip())
        if 'vflip' in aug:
            data_transforms.append(transforms.RandomVerticalFlip())
    print('data augmentations:', data_transforms)
    
    cfg.INPUT_SHAPE = (args.input_size, args.input_size)
    data_transforms.append(transforms.Resize(cfg.INPUT_SHAPE))
    transform = transforms.Compose(data_transforms)

    # define the dataset for training
    dataset = CustomDataset(cfg.TRAIN_DATA_FILE, transform)
    batchSampler = CustomBatchSampler(dataset, batch_size_per_cls=2) # batch_size 48
    data_loader = DataLoader(dataset, batch_size=1, num_workers=8, batch_sampler=batchSampler, persistent_workers=True)
    
    # define the dataset for validation
    val_dataset = CustomDataset(cfg.EVAL_DATA_FILE,transform)
    val_batchSampler = CustomBatchSampler(val_dataset, batch_size_per_cls=2) # batch_size 48
    val_data_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, batch_sampler=val_batchSampler, persistent_workers=True)

    model = CAE(in_size=args.input_size, encoder_name=args.encoder, embed_size=1024, classes=24)

    cfg.STEPS_PER_EPOCH = len(dataset)//48 if args.steps_per_epoch==0 else args.steps_per_epoch
    cfg.DECAY_STEPS = cfg.STEPS_PER_EPOCH
    cfg.LR = args.lr
    cfg.EPOCHS = args.epochs

    # initialize the model
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    trainer = Trainer(model, cfg, device)
    trainer.start_train(data_loader, val_data_loader, args.save_dir, pretrained_file=args.load)


    
