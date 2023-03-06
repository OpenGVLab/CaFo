from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, template):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_clip_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/clip_keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/clip_values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/clip_keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/clip_values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

def build_dino_cache_model(cfg, dino_model, train_loader_cache):
    
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = dino_model(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/dino_keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/dino_values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/dino_keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/dino_values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

def build_clip_dalle_cache_model(cfg, clip_model, train_loader_cache):
    
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/clip_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/clip_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/clip_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/clip_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    return cache_keys, cache_values

def build_dino_dalle_cache_model(cfg, dino_model, train_loader_cache):
    
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = dino_model(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/dino_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/dino_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/dino_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/dino_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_CLIP_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_clip_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_clip_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_clip_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_clip_l.pt")
    
    return features, labels


def pre_DINO_load_features(cfg, split, dino_model, loader):
    
    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = dino_model(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_dino_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_dino_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_dino_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_dino_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

def search_no_clip_hp(cfg, cache_keys, cache_values, features, labels, adapter=None):
    
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features).to(torch.float16)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                # clip_logits = 100. * features @ clip_weights
                # tip_logits = clip_logits + cache_logits * alpha
                tip_logits = cache_logits
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def search_ensemble_hp(cfg, 
                    clip_cache_keys, 
                    clip_cache_values, 
                    clip_features, 
                    dino_cache_keys, 
                    dino_cache_values, 
                    dino_features, 
                    labels, 
                    clip_weights, 
                    clip_adapter=None, 
                    dino_adapter=None):
    
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if clip_adapter:
                    clip_affinity = clip_adapter(clip_features)
                    dino_affinity = dino_adapter(dino_features).to(dino_cache_values)
                else:
                    clip_affinity = clip_features @ clip_cache_keys
                    dino_affinity = (dino_features @ dino_cache_keys).to(dino_cache_values)

                clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
                dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
                clip_logits = 100. * clip_features @ clip_weights
                cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
        with open("best.txt","w") as f:
            f.write("After searching, the best accuarcy: {:.2f}.\n".format(best_acc))
    return best_beta, best_alpha


# clip zero_shot as baseline
def logits_fuse(zero_logtis, logits, normalize='mean'):
    # normalize logits
    softmax_fun = nn.Softmax(dim=1)
    if normalize == 'softmax':
        zero_logtis = softmax_fun(zero_logtis)
    elif normalize =='linear':
        zero_logtis /= torch.norm(zero_logtis, p=2, dim=1, keepdim=True)
    elif normalize == 'mean':
        logits_std = torch.std(zero_logtis, dim=1, keepdim=True)
        logits_mean = torch.mean(zero_logtis, dim=1, keepdim=True)
        zero_logtis = (zero_logtis - logits_mean) / logits_std
    else:
        raise("error normalize!")
    similarity_matrix = []
    normalize_logits = []
    for logit in logits:
        if normalize == 'softmax':
            current_normalize_logits = softmax_fun(logit)
        elif normalize =='linear':
            current_normalize_logits = logit / torch.norm(logit, p=2, dim=1, keepdim=True)
        elif normalize == 'mean':
            logits_std = torch.std(logit, dim=1, keepdim=True)
            logits_mean = torch.mean(logit, dim=1, keepdim=True)
            current_normalize_logits = (logit - logits_mean) / logits_std
        else:
            raise("error normalize!")
        current_similarity = current_normalize_logits * zero_logtis
        current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
        similarity_matrix.append(current_similarity)
        normalize_logits.append(current_normalize_logits)
    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    similarity_matrix = softmax_fun(similarity_matrix)
    normalize_logits = torch.stack(normalize_logits, dim=-2)
    result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)

    return result_logits
def logits_fuse_s(zero_logtis, logits, normalize='mean'):
    # normalize logits
    softmax_fun = nn.Softmax(dim=1)
    if normalize == 'softmax':
        zero_logtis = softmax_fun(zero_logtis)
    elif normalize =='linear':
        zero_logtis /= torch.norm(zero_logtis, p=2, dim=1, keepdim=True)
    elif normalize == 'mean':
        logits_std = torch.std(zero_logtis, dim=1, keepdim=True)
        logits_mean = torch.mean(zero_logtis, dim=1, keepdim=True)
        zero_logtis = (zero_logtis - logits_mean) / logits_std
    else:
        raise("error normalize!")
    similarity_matrix = []
    normalize_logits = []
    for logit in logits:
        if normalize == 'softmax':
            current_normalize_logits = softmax_fun(logit)
        elif normalize =='linear':
            current_normalize_logits = logit / torch.norm(logit, p=2, dim=1, keepdim=True)
        elif normalize == 'mean':
            logits_std = torch.std(logit, dim=1, keepdim=True)
            logits_mean = torch.mean(logit, dim=1, keepdim=True)
            current_normalize_logits = (logit - logits_mean) / logits_std
        else:
            raise("error normalize!")
        current_similarity = current_normalize_logits * zero_logtis
        current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
        similarity_matrix.append(current_similarity)
        normalize_logits.append(current_normalize_logits)
    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    similarity_matrix = softmax_fun(similarity_matrix)
    count = 0
    for i in similarity_matrix:
        if i[0]>0.4 and i[0]<0.6:
            count += 1
    normalize_logits = torch.stack(normalize_logits, dim=-2)
    result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)

    return result_logits, count
