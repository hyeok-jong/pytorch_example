



import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def set_parser():
    parser = argparse.ArgumentParser('')

    parser.add_argument('--master_port', type = str, default = '12357', help = 'port to use DDP processing')
    
    parser.add_argument('--model_name', type = str)

    parser.add_argument('--learning_method', type = str)

    parser.add_argument('--batch_size', type = int, help = 'batch size for each GPU')
    parser.add_argument('--lr', type = float)
    parser.add_argument('--epochs', type = int)
    parser.add_argument('--stop_epoch', type = int, default = False, help = 'stop epoch for dubugging or hyperparameter tuning. (for grid search runs only few epochs)')

    parser.add_argument('--optimizer', type = str)
    parser.add_argument('--scheduler', type = str)
    parser.add_argument('--weight_decay', type = float)
    
    parser.add_argument('--image_size', type = int)
    
    # parser.add_argument('--cutmix', type = str2bool)
    parser.add_argument('--mixup', type = str2bool)
    
    args = parser.parse_args()

    # set experiment name for save logs and wandb logging
    exper_name = ''
    for key, val in vars(args).items():
        print(key, val)
        exper_name += f'{key}:{val}__'
    
    args.exper_name = exper_name
    
    return args