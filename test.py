import json
from models.cae import CAE
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from cfgs.config import cfg
import argparse
import glob
import os.path as osp
import torchmetrics
import matplotlib.pyplot as plt
import csv
import os


def get_args():
    parser = argparse.ArgumentParser(description='Test the CAE for chromosome anormaly detection')
    parser.add_argument('--encoder', '-encoder', type=str, default='resnet50', help='The backbone for feature extraction,\
                        optional: resnet34, resnet50, resnet101, mit_b0, mit_b1, mit_b2,...')
    parser.add_argument('--decoder', '-decoder', type=str, default='unet', help='The structure for feature restoring,\
                        optional: unet, fpn,...')
    parser.add_argument('--im_path', '-im_path', type=str, default=r'your json file path', help='Path of images to test')
    parser.add_argument('--input_size', '-input_size', type=int, default=224, help='the feed size of image')
    parser.add_argument('--load', '-load', type=str, default='weights_best.h5', help='Load model from a .h5 file')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='resnet_unet', help='the path to save weights')
   
    return parser.parse_args()


def load_images_from_json(json_file, root_dir):
    """
    load images and labels from a JSON file.
    :param json_file: JSON file path
    :param root_dir: image root directory
    :return: image_paths, labels
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_paths = []
    labels = []

    for item in data['items']:
        if item.get("kar_type") == "normal":
            for key, patches in item.items():
                if key != "kar_files" and isinstance(patches, list):
                    for patch_info in patches:
                        relative_path = patch_info[0].strip() 
                        full_path = osp.join(root_dir, relative_path)
                        image_paths.append(full_path)
                        labels.append(patch_info[1])

    return image_paths, labels

  

if __name__ == '__main__':
    args = get_args()
    
    data_transforms = []   
    cfg.INPUT_SHAPE = (args.input_size, args.input_size)
    data_transforms.append(transforms.Resize(cfg.INPUT_SHAPE))
    transform = transforms.Compose(data_transforms)
    
    root_dir = "your image root directory" 
    json_file = args.im_path 
    im_files, labels = load_images_from_json(json_file, root_dir)

        
    # initialize the model
    model = CAE(in_size=args.input_size, encoder_name=args.encoder, embed_size=1024, classes=24)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.load_state_dict(torch.load(args.load, map_location=device))
    model.to(device=device)
    model.eval()

    total = len(im_files)
    done = 0
    preds = []
    probs = []
    embeddings = []

    for i in range(len(im_files)):
        im_path = im_files[i] 
        label = labels[i]

        base_path = osp.basename(osp.dirname(im_path))
        save_path = osp.join(args.save_dir, base_path)

        print(done, total, im_path)  
        done += 1

        im = Image.open(im_path).convert('RGB')      
        im = transform(im)
        im = np.asarray(im)       
        imorg = im.copy()        

        im = im[None, ...]/255.0
        im = torch.tensor(im, dtype=torch.float32)         
        im = im.permute(0, 3, 1, 2).to(device) 
        
        with torch.no_grad():
            embedding, pred_label, logit_scale = model(im)

        embedding = embedding.cpu().numpy()
        embeddings.append(embedding)
        
        pred_label = F.softmax(pred_label, dim=-1)
        probs.append(pred_label)
        pred_label = pred_label.cpu().numpy()[0, ...]

        pred_prob = round(pred_label[np.argmax(pred_label)], 4)
        pred_label = np.argmax(pred_label) + 1
        preds.append(pred_label)

    preds = torch.tensor(np.array(preds))
    labels = torch.tensor(np.array(labels))
    probs = torch.cat(probs, dim=0)

    # Calculate metrics
    acc = torchmetrics.functional.accuracy(preds, labels, 'multiclass', num_classes=24)
    print('accuracy:', acc.cpu().numpy())
    preds = preds - 1
    labels = labels - 1
    recall = torchmetrics.functional.recall(preds, labels, average='macro', num_classes=24, task='multiclass')
    print('Recall:', recall.cpu().numpy())
    labels = labels.to(probs.device)
    auc = torchmetrics.functional.auroc(probs, labels, num_classes=24, task='multiclass')
    print('AUC:', auc.cpu().numpy())

    # Save metrics to CSV
    file_name = 'test.csv'
    metrics = {'Accuracy': acc.cpu().numpy(), 'Recall': recall.cpu().numpy(), 'AUC': auc.cpu().numpy()}
    with open(args.save_dir + file_name, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Metric', 'Value'])
        writer.writeheader()
        for metric, value in metrics.items():
            writer.writerow({'Metric': metric, 'Value': value})
    print(f'Metrics saved to {args.save_dir}{file_name}')

    
    


