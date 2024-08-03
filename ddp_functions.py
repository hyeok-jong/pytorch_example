import os
import torch
import torch.distributed as dist
import datetime

''' 
Error : 
E ProcessGroupNCCL.cpp:474] [Rank 6] Watchdog caught collective operation timeout: 
WorkNCCL(SeqNum=57, OpType=ALLGATHER, NumelIn=191, NumelOut=1528, Timeout(ms)=200000) 
ran for 200059 milliseconds before timing out.

찾아보면 https://github.com/huggingface/accelerate/issues/314
가끔가다 이러한 오류들이 발생하게 되는데, debugging 해본결과는 다음과 같다.

1. train function에서 특정 rank에 sleep을 300sec로 건다.
2. timeout을 200sec 으로 설정한다.
그렇게 하면 다른 GPU에서는 이미 연산이 끝났지만, 특정 gpu에서는 연산이 안끝나서 계속 기다려야 한다.
그 기다라는 시간이 timeout이고 그것보다 오래 기다리게 되면 위의 오류가 뜨게 되는것이다.

기다려야 하는 이유는 다음과 같다.
mp.spawn으로 여러개의 gpu에 function을 뿌려 주는데, 같은 에폭은 같은 시간에 끝나서 DDP를 통해 backpropagation이 동기화 되어야 한다.
따라서 하나만 늦게 되도 전체적으로 지연되는 것이다.
특히 보통 rank == 0일때 acc를 계산하거나, 모델을 저장하거나 하는데, 이러한 이유로 특정 GPU에서만 시간이 default인 10min보다 느리게 되면 오류가 발생한다.
물론 10min을 넘는 경우는 없을거 같긴한데, 나는 몇번 중간에 위의 오류가 발생했었다. (학습 초반에는 정상이었다가 몇시간 있다가 자꾸 오류 발생했고, 아마 특정 GPU가 과부하 걸려서 느려진것 같았음.)

또한 통신시간이 오래 걸리는 것도 이유일 수 있다.

따라서 이를 해결하기 위해, timeout을 충분히 길게 해줬다.

또다른 방법으로는 매 에폭마다 dist.barrier()를 해주는 것이 있는듯 한데, 위의 경우에서는 안되었음. 그냥 timeout 늘리는 것이 답인듯
'''


def setup(rank, world_size, master_port = '12357'):
    '''
    Occasionally, forcibly quitting a process can result in the master port not being released.
    Restarting Docker can be an effective way to free up the port.
    Alternatively, you can use net-tools to manually release the port:
    1. Update the package list:
    sudo apt update
    2. Install net-tools:
    sudo apt install net-tools
    3. Find the process ID (PID) using the port (e.g., port 12357):
    netstat -nlp | grep :12357
    4. Kill the process using the PID:
    sudo kill -9 {PID}
    '''
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = master_port

    # https://github.com/ultralytics/ultralytics/issues/1439
    os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
    dist.init_process_group(
        backend='nccl' if dist.is_nccl_available() else 'gloo', 
        timeout=datetime.timedelta(seconds=7200000),
        # timeout=datetime.timedelta(seconds=200),
        rank=rank, 
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def gather_tensors(tensor, rank, target_rank=0):
    # Create a list of tensors with the same shape as the input tensor, one for each process
    gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]

    # dist.all_gather must be called from all ranks
    dist.all_gather(gather_list, tensor)

    # return cocatenated tensor only if target rank.
    if rank == target_rank:
        return torch.cat(gather_list, dim=0)
    return None


def set_dataloader(dataset, world_size, rank, batch_size, num_workers, pin_memory, return_sampler=False):
    # The sampler uses the total number of GPUs (world_size) and the current rank.
    # The sampler takes the original dataset, divides it according to world_size, 
    # and selects the appropriate subset for the current rank.
    dataset_sampler = torch.utils.data.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=dataset_sampler
    )
    # During training, the dataset_sampler should shuffle the dataset to ensure -> see main.py line of train_sampler.set_epoch(epoch)
    # that each rank gets a different random subset each epoch.
    if return_sampler:
        return data_loader, dataset_sampler
    else:
        return data_loader
