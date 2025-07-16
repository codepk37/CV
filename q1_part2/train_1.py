import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader
import wandb
import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.anchor_utils import AnchorGenerator
from metrices import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import csv
def log_to_csv(data, filename="st_1.csv"):
    # Check if the file exists, to decide whether to write headers
    file_exists = False
    try:
        with open(filename, mode='r'):
            file_exists = True
    except FileNotFoundError:
        pass
    
    # Open the file in append mode
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write headers only if the file does not exist
        if not file_exists:
            writer.writerow([
                "RPN Classification Loss",
                "RPN Localization Loss",
                "FRCNN Classification Loss",
                "FRCNN Localization Loss",
                "FRCNN Angle Loss",
                "Total Loss",
                "MAP Score"
            ])
        
        # Write the data
        writer.writerow(data)


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    wandb.init(project="CV_1 st_1.yaml")
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    st = SceneTextDataset('train', root_dir=dataset_config['root_dir'])
    print("st ",st)
    train_dataset = DataLoader(st,
                               batch_size=4,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=collate_function)


    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                        min_size=600,
                                                        max_size=1000,
                                                        angle_binsize=train_config['angle_binsize'] ## 0 menas regression , >= 1 refers to classification
    )

    
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(  #set it k,4*k
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes'] ,angle_binsize=train_config['angle_binsize'])
    
    if False:
        checkpoint_path = os.path.join(train_config['task_name'], 'tv_frcnn_r50fpn_' + train_config['ckpt_name'])
        faster_rcnn_model.load_state_dict(torch.load(checkpoint_path))
    print(faster_rcnn_model)

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.SGD(lr=1E-4,
                                params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
                                weight_decay=5E-5, momentum=0.9)

    num_epochs = train_config['num_epochs']
    step_count = 0

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        frcnn_angle_loss = []
        for ims, targets, _ in tqdm(train_dataset):
            optimizer.zero_grad()
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
                target['angles'] = target['angles'].float().to(device) #info of angle theta is given

                # print(target['angles'],"target['angles'] ")

            images = [im.float().to(device) for im in ims]
            # print(target,"target ") 
            # {'labels': tensor([1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0'), 'boxes': tensor([[ 586.0000,  487.0000,  737.0000,  525.0000],
            # [ 132.1559,  203.9439,  731.1569,  325.2668],
            # [ 730.6308,  192.1403,  823.5781,  285.0198],
            # [ 826.4935,  210.8728, 1186.3949,  320.7057],
            # [ 306.6217,  370.0024,  899.4325,  429.9834],
            # [ 438.8139,  482.4992,  555.8182,  525.5319],
            # [1187.5499,  481.3250, 1250.4506,  533.6962],
            # [1061.6855,  483.2879, 1092.7017,  501.4840]], device='cuda:0')} target 
            batch_losses = faster_rcnn_model(images, targets) 
            loss = batch_losses['loss_classifier']
            loss += batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']
            loss += batch_losses['loss_angle'] * train_config['angle_weight']
            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())
            frcnn_angle_loss.append(batch_losses['loss_angle'].item())

            loss.backward()
            optimizer.step()
            step_count +=1
        print('Finished epoch {}'.format(i))
        torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                'tv_frcnn_r50fpn_' + train_config['ckpt_name']))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        loss_output += ' | FRCNN Angle Loss : {:.4f}'.format(np.mean(frcnn_angle_loss))
        print(loss_output)
        loss_number  = np.mean(frcnn_classification_losses) + np.mean(frcnn_localization_losses) + np.mean(frcnn_angle_loss) + np.mean(rpn_classification_losses) + np.mean(rpn_localization_losses)


        a= evaluate_map('config/st_1.yaml')
        print(a)

        wandb.log({"RPN Classification Loss":np.mean(rpn_classification_losses),
                    "RPN Localization Loss":np.mean(rpn_localization_losses),
                    "FRCNN Classification Loss":np.mean(frcnn_classification_losses),
                    "FRCNN Localization Loss":np.mean(frcnn_localization_losses),
                    "FRCNN Angle Loss":np.mean(frcnn_angle_loss),
                    "total loss":loss_number,
                    "MAP Score":a
                    })
        
        log_data = [
            np.mean(rpn_classification_losses),
            np.mean(rpn_localization_losses),
            np.mean(frcnn_classification_losses),
            np.mean(frcnn_localization_losses),
            np.mean(frcnn_angle_loss),
            loss_number,
            a
        ]
        log_to_csv(log_data)

        print("total loss ",loss_number)
    wandb.finish()
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path',
                        default='config/st_1.yaml', type=str)
    args = parser.parse_args()
    train(args)