import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


def no_of_corrects(output, label, threshold = 0.5):
    prediction = output.detach().argmax(dim = -1)
    corrects = (label == prediction).sum()
    return corrects
    



def get_ddp_results(result_dict, prefix, rank, world_size):
    return_dict = dict()
    for key, val in result_dict.items():
        if isinstance(val, torch.Tensor):
            metric_tensor = val.clone().detach().to(dtype=torch.float32, device=rank)
        else:
            metric_tensor = torch.tensor(val, dtype=torch.float32).to(device=rank)
        return_list = [torch.zeros(1, device = rank, dtype = torch.float32) for _ in range(world_size)]
        torch.distributed.all_gather(return_list, metric_tensor)
        metric = sum(return_list)/len(return_list)
        return_dict[f'{prefix}{key}'] = metric.cpu().numpy()
        
    return return_dict
    
    
@torch.no_grad()
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
    
def set_random(random_seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)





# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal

def color_print(text, color='red', bg_color=None, bold=False, underline=False, end = None):
    # make all types to string.
    text = f'{text}'
    color_dict = {
        'black': '\033[30m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'orange': '\033[33m',
        'purple': '\033[35m',
        'pink': '\033[95m',
        'white': '\033[97m'
    }

    bg_color_dict = {
        'black': '\033[40m',
        'red': '\033[41m',
        'green': '\033[42m',
        'yellow': '\033[43m',
        'blue': '\033[44m',
        'magenta': '\033[45m',
        'cyan': '\033[46m',
        'white': '\033[47m',
    }

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    STRIKETHROUGH = '\033[09m'
    ENDC = '\033[0m'

    if bold:
        text = BOLD + text
    if underline:
        text = UNDERLINE + text
    if color:
        text = color_dict[color.lower()] + text
    if bg_color:
        text = bg_color_dict[bg_color.lower()] + text
    print(text + ENDC, end = end)