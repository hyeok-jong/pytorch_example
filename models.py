
'''
bn을 조심해야한다.
bn은 requires_grad == False여도 .train()으로 하면 running_mean과 running_var이 바뀌어 .eval()시에 바뀌게 된다.
즉, eval()시에는 running_mean, running_var사용하기 때문이다.
따라서, freeze부분은 그냥 .eval로 해두는 것이 좋다.

optimzier에는 모두 들어가도 상관 없음.
즉, optimizer에서는 requires_grad == True인 부분만 바뀌긴 하지만, 불필요한 연산을 할 수도 있긴함.

쉽게 생각하면,
eval / train은 forward시에 바뀌는거고, requires_grad는 backward에서 바뀌는것.
'''

import torch
from torchvision import models
import timm
import loralib as lora

from utils import color_print

dinov2_dict = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

clip_dict = {
    'vit_giant_patch14_clip_224.laion2b': 1408,
    'vit_large_patch14_clip_224.laion2b': 1024,
    'vit_base_patch16_clip_224.laion2b': 768,
}

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False



def set_model(model_name, rank, learning_method, verbose = True, out_features = 10):

    
    model_name = model_name.lower()
    
    if model_name == 'convnext_large':
        model = models.convnext_large(weights = models.ConvNeXt_Large_Weights.DEFAULT)
        model.classifier[-1] = torch.nn.Linear(in_features = 1536, out_features = out_features, bias = True)

        if learning_method == 'lora':
            init_model = model.state_dict()

            for name, module in model.named_modules():
                if type(module).__name__ == 'Conv2d':
                    conv_lora = lora.ConvLoRA(
                        conv_module = torch.nn.Conv2d,
                        in_channels = module.in_channels,
                        out_channels = module.out_channels,
                        kernel_size = module.kernel_size[0],
                        r = 4, 
                        lora_alpha = 4,
                        merge_weights = False,
                        stride = module.stride,
                        padding = module.padding,
                        dilation = module.dilation,
                        groups = module.groups,
                        bias = module.bias is not None,
                        padding_mode = module.padding_mode
                    )
                    
                    parent_name = name.rsplit('.', 1)[0]  
                    parent_module = dict(model.named_modules())[parent_name]
                    setattr(parent_module, name.split('.')[-1], conv_lora)
                    
            false_keys = model.load_state_dict(init_model, strict = False)
            lora.mark_only_lora_as_trainable(model)
            unfreeze(model.classifier[-1])
            unfreeze(model.classifier[-3])



    elif 'dinov2' in model_name:
        model_name = 'dinov2_vits14'
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        model.head = torch.nn.Linear(in_features = dinov2_dict[model_name], out_features = out_features, bias = True)
        if learning_method == 'classifier':
            freeze(model)
            unfreeze(model.norm)
            unfreeze(model.head)
        elif learning_method == 'partial':
            freeze(model)
            unfreeze(model.norm)
            unfreeze(model.head)
            unfreeze(model.blocks[-1])
        elif learning_method == 'lora':
            init_model = model.state_dict()
            for _, blk in enumerate(model.blocks):
                in_features = blk.attn.qkv.in_features
                out_features = blk.attn.qkv.out_features
                blk.attn.qkv =  lora.MergedLinear(in_features, out_features, r = 4, lora_alpha = 4, enable_lora = [True, False, True], merge_weights = False)
            # MergedLinear가 nn.Linear를 상속 받으면서 random initialization된다. 따라서 다시 load해줘야 한다.
            false_keys = model.load_state_dict(init_model, strict = False)

            lora.mark_only_lora_as_trainable(model)
            unfreeze(model.norm)
            unfreeze(model.head)
            print(false_keys)
            
    elif 'openclip' in model_name:
        model_name = 'vit_giant_patch14_clip_224.laion2b'
        model = timm.create_model(model_name, pretrained = True, num_classes = out_features, img_size = 224)
        if learning_method == 'classifier':
            freeze(model)
            unfreeze(model.norm)
            unfreeze(model.head)
        elif learning_method == 'partial':
            freeze(model)
            unfreeze(model.norm)
            unfreeze(model.head)
            unfreeze(model.blocks[-1])

        elif learning_method == 'lora':
            init_model = model.state_dict()
            for _, blk in enumerate(model.blocks):
                in_features = blk.attn.qkv.in_features
                out_features = blk.attn.qkv.out_features
                blk.attn.qkv =  lora.MergedLinear(in_features, out_features, r = 4, lora_alpha = 4, enable_lora = [True, False, True], merge_weights = False)
            # MergedLinear가 nn.Linear를 상속 받으면서 random initialization된다. 따라서 다시 load해줘야 한다.
            false_keys = model.load_state_dict(init_model, strict = False)

            lora.mark_only_lora_as_trainable(model)
            unfreeze(model.norm)
            unfreeze(model.head)
            print(false_keys)


    elif 'mae' in model_name:
        # https://huggingface.co/timm/vit_huge_patch14_224.mae
        # search model on https://huggingface.co/timm?search_models=.mae 
        model = timm.create_model('vit_large_patch16_224.mae', pretrained = True, num_classes = out_features, img_size = 224)
        if learning_method == 'classifier':
            freeze(model)
            unfreeze(model.norm)
            unfreeze(model.head)
        elif learning_method == 'partial':
            freeze(model)
            unfreeze(model.norm)
            unfreeze(model.head)
            unfreeze(model.blocks[-1])

        elif learning_method == 'lora':
            init_model = model.state_dict()
            for _, blk in enumerate(model.blocks):
                in_features = blk.attn.qkv.in_features
                out_features = blk.attn.qkv.out_features
                blk.attn.qkv =  lora.MergedLinear(in_features, out_features, r = 4, lora_alpha = 4, enable_lora = [True, False, True], merge_weights = False)
            # MergedLinear가 nn.Linear를 상속 받으면서 random initialization된다. 따라서 다시 load해줘야 한다.
            false_keys = model.load_state_dict(init_model, strict = False)

            lora.mark_only_lora_as_trainable(model)
            unfreeze(model.norm)
            unfreeze(model.head)
            print(false_keys)

    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_parameters = sum(p.numel() for p in model.parameters())
        print('*'*100)
        print('no of all parameters :', format(all_parameters, ','))
        print('no of trainable parameters :', format(trainable, ','))
        print('trainable ratio :', round(trainable / all_parameters * 100, 3), '%')
        print('*'*100)

        if verbose:
            for name, params in model.named_parameters():
                if params.requires_grad:
                    color_print(f'{name} \U0001F3C3 ready to update', 'red')
                else:
                    color_print(f'{name} \U0001F9D8 not gonna move', 'blue')

        
    return model.cuda(rank)
        