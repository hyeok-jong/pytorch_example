import torch
import numpy as np
from tqdm import tqdm
from ddp_functions import gather_tensors
from CAWR import CosineAnnealingWarmUpRestarts
import time
from utils import no_of_corrects



def train(dataloader, model, loss_function, optimizer, rank, epoch, args_parser, verbose = True):
    model.train()

    # Example: Metrics like AUC must be computed using the gathered tensors 
    # rather than computing them separately on each rank and then gathering, unlike accuracy.
    output_list = list()
    label_list = list()
    corrects_list = list()
    loss_list = list()
    
    if verbose and rank == 0:
        progress_bar = tqdm(total=len(dataloader), desc=f'Epoch {epoch} Rank {rank}')
    for i, data in enumerate(dataloader):
        image, label = data
        image = image.cuda(rank)
        label = label.cuda(rank)
        batch_size = label.shape[0]

        # for debugging
        if epoch == 0 and i == 0 and rank == 0:
            print(image.shape)
            print(label.shape)
            
        if args_parser.mixup:
            image, label, shuffled_label, lam = mixup(image, label)

        output = model(image)
        
        if args_parser.mixup:
            loss = lam * loss_function(output, label) + (1 - lam) * loss_function(output, shuffled_label)
            corrects = lam * no_of_corrects(output, label) + (1 - lam) * no_of_corrects(output, shuffled_label)
        else:
            loss = loss_function(output, label)
            corrects = no_of_corrects(output, label)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output_list.append(output.detach())
        label_list.append(label)
        corrects_list.append(corrects.cuda(rank))
        loss_list.append(batch_size * loss)
        

        if verbose and rank == 0:
            progress_bar.update(1)
        
    all_outputs = torch.cat(output_list)
    all_labels = torch.cat(label_list)
    all_corrects = torch.stack(corrects_list)
    all_losses = torch.stack(loss_list)
    
    # 모든 rank에서 호출 되어야 하지만, 실제 값은 rank==0에서만 받음.
    gathered_outputs = gather_tensors(all_outputs, rank)
    gathered_labels = gather_tensors(all_labels, rank)
    gathered_corrects = gather_tensors(all_corrects, rank)
    gathered_losses = gather_tensors(all_losses, rank)

    print(f'rank : {rank} | epoch : {epoch} done!')

    # see ddp_functions.py
    # if rank == 0:
    #     time.sleep(300)
    
    return gathered_outputs, gathered_labels, gathered_corrects, gathered_losses


@torch.no_grad()
def inference(dataloader, model, loss_function, rank, epoch, args_parser):
    # inference에서는 사실 single로 돌리는게 편함.
    # 즉, dataloader 자체를 sampler 없이
    model.eval()
        
    output_list = list()
    label_list = list()
    corrects_list = list()
    loss_list = list()

    for data in dataloader:
        image, label = data
        image = image.cuda(rank)
        label = label.cuda(rank)

        batch_size = label.shape[0]

        output = model(image)
        loss = loss_function(output, label)
        corrects = no_of_corrects(output, label)

        output_list.append(output.detach())
        label_list.append(label)
        corrects_list.append(corrects.cuda(rank))
        loss_list.append(batch_size * loss)
        
        
    all_outputs = torch.cat(output_list)
    all_labels = torch.cat(label_list)
    all_corrects = torch.stack(corrects_list)
    all_losses = torch.stack(loss_list)
    
    gathered_outputs = gather_tensors(all_outputs, rank)
    gathered_labels = gather_tensors(all_labels, rank)
    gathered_corrects = gather_tensors(all_corrects, rank)
    gathered_losses = gather_tensors(all_losses, rank)
    
    return gathered_outputs, gathered_labels, gathered_corrects, gathered_losses



def set_optimizer(optimizer_name, parameters, learning_rate, weight_decay = 0.0):
    optimizer_name = optimizer_name.lower()
    trainable_parameters = [params for params in parameters if params.requires_grad]
    if optimizer_name == 'sgd':
        return torch.optim.SGD(params = trainable_parameters, lr = learning_rate, weight_decay = weight_decay)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(params = trainable_parameters, lr = learning_rate, betas = (0.9, 0.999), weight_decay = weight_decay)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(params = trainable_parameters, lr = learning_rate, weight_decay = weight_decay)


def set_lr_scheduler(optimizer, epochs, learning_rate, name):
    if name == 'CAWR':
        cycles = 2
        lr_scheduler = CosineAnnealingWarmUpRestarts(
            optimizer = optimizer, 
            T_0 = epochs // cycles, 
            T_mult = 1, 
            eta_max = learning_rate, 
            # T_up = epochs//10,
            T_up = 5,
            gamma = 0.5
            )
    elif name == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer = optimizer, 
            milestones = [60, 80], 
            gamma = 0.5, 
            last_epoch = -1
            )
    return lr_scheduler
    
    
def mixup(data, target, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    
    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    return new_data, target, shuffled_target, lam

# def cutmix(data, target, alpha=1.0):
#     indices = torch.randperm(data.size(0))
#     shuffled_data = data[indices]
#     shuffled_target = target[indices]
    
#     lam = np.random.beta(alpha, alpha)
#     bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
#     new_data = data.clone()
#     new_data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
#     new_target = target * lam + shuffled_target * (1 - lam)
    
#     return new_data, new_target

# def rand_bbox(size, lam):
#     W = size[2]
#     H = size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = int(W * cut_rat)
#     cut_h = int(H * cut_rat)

#     cx = np.random.randint(W)
#     cy = np.random.randint(H)

#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)

#     return bbx1, bby1, bbx2, bby2
