import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchvision_models

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import dino.utils as utils
import itertools
import json


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

def run_ensemble_tip_dalle_adapter_F(cfg, 
                            clip_cache_keys, 
                            clip_cache_values, 
                            clip_test_features, 
                            dino_cache_keys, 
                            dino_cache_values, 
                            dino_test_features, 
                            test_labels, 
                            clip_weights, 
                            clip_model, 
                            dino_model, 
                            train_loader_F,
                            dalle_train_loader_F):
    
    # Enable the cached keys to be learnable
    clip_adapter = nn.Linear(clip_cache_keys.shape[0], clip_cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    clip_adapter.weight = nn.Parameter(clip_cache_keys.t())
    dino_adapter = nn.Linear(dino_cache_keys.shape[0], dino_cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    dino_adapter.weight = nn.Parameter(dino_cache_keys.t())
    
    optimizer = torch.optim.AdamW(
        itertools.chain(dino_adapter.parameters(), clip_adapter.parameters()),
        lr=cfg['lr'], 
        eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        clip_adapter.train()
        dino_adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        # origin image
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                clip_image_features = clip_model.encode_image(images)
                clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                dino_image_features = dino_model(images)
                dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)

            clip_affinity = clip_adapter(clip_image_features)
            clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
            dino_affinity = dino_adapter(dino_image_features).to(dino_cache_values.dtype)
            dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
            clip_logits = 100. * clip_image_features @ clip_weights

            cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
            tip_logits = clip_logits + cache_logits * alpha
            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # dalle image
        for i, (images, target) in enumerate(tqdm(dalle_train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                clip_image_features = clip_model.encode_image(images)
                clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                dino_image_features = dino_model(images)
                dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)

            clip_affinity = clip_adapter(clip_image_features)
            clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
            dino_affinity = dino_adapter(dino_image_features).to(dino_cache_values.dtype)
            dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
            clip_logits = 100. * clip_image_features @ clip_weights

            cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
            tip_logits = clip_logits + cache_logits * alpha
            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        clip_adapter.eval()
        dino_adapter.eval()

        clip_affinity = clip_adapter(clip_test_features)
        dino_affinity = dino_adapter(dino_test_features).to(dino_cache_values.dtype)
        clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
        dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
        clip_logits = 100. * clip_test_features @ clip_weights
        cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** CaFo's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(clip_adapter.weight, cfg['cache_dir'] + "/best_F_clip_adapter_" + str(cfg['shots']) + "shots.pt")
            torch.save(dino_adapter.weight, cfg['cache_dir'] + "/best_F_dino_adapter_" + str(cfg['shots']) + "shots.pt")
    
    clip_adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_clip_adapter_" + str(cfg['shots']) + "shots.pt")
    dino_adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_dino_adapter_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, CaFo's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    del clip_logits, tip_logits, cache_logits, clip_cache_logits, dino_cache_logits, clip_affinity, dino_affinity 
    # Search Hyperparameters
    # _ = search_hp(cfg, affinity, clip_cache_values, clip_test_features, test_labels, clip_weights, clip_adapter=adapter)
    best_beta, best_alpha = search_ensemble_hp(cfg, clip_cache_keys, clip_cache_values, clip_test_features, dino_cache_keys, dino_cache_values, dino_test_features, test_labels, clip_weights, clip_adapter=clip_adapter, dino_adapter=dino_adapter)
    clip_affinity = clip_adapter(clip_test_features)
    dino_affinity = dino_adapter(dino_test_features).to(dino_cache_values.dtype)
    clip_cache_logits = ((-1) * (best_beta - best_beta * clip_affinity)).exp() @ clip_cache_values
    dino_cache_logits = ((-1) * (best_beta - best_beta * dino_affinity)).exp() @ dino_cache_values
    clip_logits = 100. * clip_test_features @ clip_weights
    cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
    tip_logits = clip_logits + cache_logits * best_alpha
    print("save logits!!!!!!!!!!!!!")
    torch.save(tip_logits, cfg['cache_dir'] + "/best_tip_dino_dalle_logits_" + str(cfg['shots']) + "shots.pt")

def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['clip_backbone'])
    clip_model.eval()

    # DINO
    dino_model = torchvision_models.__dict__[cfg['dino_backbone']](num_classes=0)
    dino_model.fc = nn.Identity()
    dino_model.cuda()
    utils.load_pretrained_weights(dino_model, "dino/dino_resnet50_pretrain.pth", "teacher", "vit_small'", 16)
    dino_model.eval()

    # ImageNet dataset
    random.seed(2)
    torch.manual_seed(1)
    
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    dalle_dataset = build_dataset(cfg['dalle_dataset'], cfg['root_path'], cfg['dalle_shots'])
    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    dalle_train_loader_cache = build_data_loader(data_source=dalle_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    dalle_train_loader_F = build_data_loader(data_source=dalle_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)
    
    with open(cfg['gpt3_prompt_file']) as f:
        gpt3_prompt = json.load(f)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = gpt_clip_classifier(imagenet.classnames, gpt3_prompt, clip_model, imagenet.template)
    

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    print("\nConstructing CLIP cache model.")
    clip_cache_keys, clip_cache_values = build_clip_cache_model(cfg, clip_model, train_loader_cache)
    print("\nConstructing DINO cache model.")
    dino_cache_keys, dino_cache_values = build_dino_cache_model(cfg, dino_model, train_loader_cache)

    print("\nConstructing cache model by dalle image.")
    print("\nConstructing CLIP cache model.")
    clip_dalle_cache_keys, clip_dalle_cache_values = build_clip_dalle_cache_model(cfg, clip_model, dalle_train_loader_cache)
    print("\nConstructing DINO cache model.")
    dino_dalle_cache_keys, dino_dalle_cache_values = build_dino_dalle_cache_model(cfg, dino_model, dalle_train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    print("\nLoading CLIP feature.")
    test_clip_features, test_labels = pre_CLIP_load_features(cfg, "test", clip_model, test_loader)
    print("\nLoading DINO feature.")
    test_dino_features, test_labels = pre_DINO_load_features(cfg, "test", dino_model, test_loader)
    
    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
   
    run_ensemble_tip_dalle_adapter_F(cfg, 
                            torch.cat((clip_cache_keys, clip_dalle_cache_keys), dim=1),
                            torch.cat((clip_cache_values, clip_dalle_cache_values), dim=0), 
                            test_clip_features, 
                            torch.cat((dino_cache_keys, dino_dalle_cache_keys), dim=1), 
                            torch.cat((dino_cache_values, dino_dalle_cache_values), dim=0), 
                            test_dino_features, 
                            test_labels, 
                            clip_weights, 
                            clip_model, 
                            dino_model, 
                            train_loader_F,
                            dalle_train_loader_F)

if __name__ == '__main__':
    main()