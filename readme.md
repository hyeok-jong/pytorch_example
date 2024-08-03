# pytorch baseline code for deep learning trianing.  

## 1. Features.  
1. [DDP (Distributed Data Parallel)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for multi-gpu training.  
2. [Grid-search script](https://github.com/hyeok-jong/grid_generator).  
3. [wandb](https://wandb.ai/home) for logging and hyperparameter importance.  
4. [vscode-server](https://github.com/coder/code-server) with [Docker](https://www.docker.com/) environment. [Dockerfile](https://github.com/hyeok-jong/Dockerfiles/tree/main/vscode_web).  
5. Various training method. Full training, Partial finetuning, and [LoRA](https://github.com/microsoft/LoRA).  
6. [Cosine Annealing Warmup Restart](https://gaussian37.github.io/dl-pytorch-lr_scheduler/) learning rate scheduler.  
7. [Colored print](https://github.com/hyeok-jong/color_print) for terminal.  
8. Other feauters annotated on py files.  



## 2. TO DO.  
1. [XAI](https://github.com/jacobgil/pytorch-grad-cam/tree/master).  
2. Add new branch or option for [DP (Data Parallel)](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) and naive training (no multi-gpu).  
3. [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).  
4. [compile models](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).  
5. Add Cutmix.  
6. More blocks partial finetuning.  