import time
import random
import math
import os
import pickle as pk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import mplfinance as mpf

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

import warnings
warnings.filterwarnings('ignore')

def make_voc(intervals):

    voc = np.array([])

    for inter in intervals:
        voc = np.concatenate((voc, np.linspace(*inter)[:-1]))

    voc = np.append(voc, intervals[-1][1])

    return voc


def inv(x):
    return 1/(x+1) - 1


VOCABULARY_INTERVALS = [
            [
                [inv(2), inv(1), 10+1], [inv(1), inv(0.1), 90+1],[inv(0.1), 0, 1000+1],
                [0, 0.1, 1000+1], [0.1, 1, 90+1], [1, 2, 10+1]
            ], 
            [
                [0, 0.1, 1000+1], [0.1, 1, 90+1], [1, 2, 10+1]
            ], 
            [
                [inv(2), inv(1), 10+1], [inv(1), inv(0.1), 90+1],[inv(0.1), 0, 1000+1]
            ],
            [
                [inv(100), inv(10), 90+1], [inv(10), inv(2), 80+1], [inv(2), inv(0.1), 190+1], [inv(0.1), inv(0), 1000+1],   
                [0, 0.1, 1000+1], [0.1, 2, 190+1], [2,10, 80+1], [10,100, 90+1]
            ], 
            [
                [inv(100), inv(10),90+1], [inv(10), inv(1), 90+1], [inv(1), inv(0), 100+1],
                [0, 1, 100+1], [1, 10, 90+1], [10, 100, 90+1]
            ],
            [
                [inv(100), inv(10),90+1], [inv(10), inv(1), 90+1], [inv(1), inv(0), 100+1],
                [0, 1, 100+1], [1, 10, 90+1], [10, 100, 90+1]
            ],
            [
                [inv(100), inv(10),90+1], [inv(10), inv(1), 90+1], [inv(1), inv(0), 100+1],
                [0, 1, 100+1], [1, 10, 90+1], [10, 100, 90+1]
            ],
            [
                [inv(100), inv(10),90+1], [inv(10), inv(1), 90+1], [inv(1), inv(0), 100+1],
                [0, 1, 100+1], [1, 10, 90+1], [10, 100, 90+1]
            ],
             ]


def build_optimizer(model, lr, wd, corr_weight_decay):

    if corr_weight_decay:

        decay = set()
        no_decay = set()

        for name, param in model.named_parameters():
            if 'bias' in name or 'norm' in name :
                no_decay.add(name)
            else:
                decay.add(name)

        # Build optimizer groups
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay],
                "weight_decay": wd,
            },
            {
                "params": [p for n, p in model.named_parameters() if n in no_decay],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    return optimizer


def get_linear_warmup_cosine_decay_scheduler(optimizer, total_steps, warmup):
    
    warmup_steps = total_steps * warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 0.99)))
        
    
    return LambdaLR(optimizer, lr_lambda)


def count_batches(dir, sequence, batch):

    batches_per_epoch = 0
    datasets = np.array(os.listdir(dir))

    for i in range(len(datasets)):

        with open(f'{dir}/{datasets[i]}', 'rb') as file:
            train = pk.load(file)

        batches_per_epoch += len(train) // (sequence+1)

    batches_per_epoch = np.ceil(batches_per_epoch / batch).astype(int)

    return batches_per_epoch


def loader(args, train=True):

    if train:
        dir = args.data_train
        batches_per_epoch = args.train_batches_per_epoch
    else:
        dir = args.data_test
        batches_per_epoch = args.test_batches_per_epoch
    
    num_features = len(args.voc_size)
    sequence = args.sequence
    batch = args.batch

    ##################list of all datasets###################
    datasets = np.array(os.listdir(dir))
    np.random.shuffle(datasets)

    #########make a queue, where we push datasets on a need basis########
    queue = np.array([]).reshape(-1, num_features)

    
    length = (sequence+1)*batch

    while (length <= len(queue) or len(datasets) != 0) and (batches_per_epoch > 0):


        if len(queue) < length and len(datasets) != 0:

            with open(f'{dir}/{datasets[0]}', 'rb') as file:
                data = pk.load(file)

            datasets = np.delete(datasets, [0])

            samples_ind = np.random.randint(0, len(data) - (sequence+1), (len(data) // (sequence+1),))
            queue = np.concatenate((queue, *[data[i : i + (sequence+1)] for i in samples_ind]))

        if length <= len(queue):

            actual_batch = min(batch, len(queue) // (sequence+1))
            x_patt = np.array([True]*sequence + [False])
            y_patt = np.array([False] + [True]*sequence)

            x = queue[ : actual_batch * (sequence+1)][np.tile(x_patt, actual_batch)].reshape(-1 ,sequence, num_features)
            y = queue[ : actual_batch * (sequence+1)][np.tile(y_patt, actual_batch)].reshape(-1 ,sequence, num_features)

            x = torch.LongTensor(x)
            y = torch.LongTensor(y)

            queue = np.delete(queue, slice(0, length), axis=0)

            batches_per_epoch -= 1

            yield x, y


def vocab_to_rel(train):

    relative = np.empty_like(train).astype(np.float64)

    for i in range(relative.shape[-1]):

        vocab = make_voc(VOCABULARY_INTERVALS[i])

        relative[:,:,i] = vocab[train[:,:,i]]

    return relative


def relative_to_abs(train, start_data):

    train = train.copy()

    ##############return back to abs values form relative####################


    abs_train = pd.DataFrame(index=train.index,
                             columns=['open', 'high', 'low', 'close', 'volume', 'v_trades'])

    ##############open high low close#############################

    bars = train.loc[:, ['open', 'high', 'low', 'close']].values + 1

    bars[0,0] *= start_data['open']
    bars[0,3] *= bars[0,0]
    bars[0,1] *= np.max((bars[0,0], bars[0,3]))
    bars[0,2] *= np.min((bars[0,0], bars[0,3]))

    for i in range(1, len(bars)):
        bars[i,0] *= bars[i-1,3]
        bars[i,3] *= bars[i,0]
        bars[i,1] *= np.max((bars[i,0], bars[i,3]))
        bars[i,2] *= np.min((bars[i,0], bars[i,3]))


    abs_train[['open', 'high', 'low', 'close']] = bars


    ##############volume and average volume per transaction####################
    a = 2 / (50 + 1)

    for col in ['volume', 'v_trades']:
        
        data = np.ones(len(abs_train)) * start_data[col]
        ema = np.ones(len(abs_train)) * start_data[col]

        ref = train[f'{col}_ema_50'] + 1

        

        for i in range(1, len(data)):
            ema[i] = ((1-a) * ema[i-1]) / (1 - a * ref[i])
            data[i] = ref[i] * ema[i]


        abs_train[col] = data

    return abs_train


def candle_plot(data, title):

    ###########выводим график и дополнительные индикаторы#####################

    data.index = pd.date_range(start='2000-01-01', periods=len(data), freq='D')

    mc = mpf.make_marketcolors(up='#f2dccc',down='#716f6f',volume='inherit')
    s  = mpf.make_mpf_style(base_mpf_style='default', marketcolors=mc)

    colors = ['#f2dccc' if c >= o else '#716f6f' for c, o in zip(data['close'], data['open'])]

    ap = mpf.make_addplot(
        data['v_trades'],
        type='bar',
        panel=2,          
        color=colors,
        width = 1

    )
    fig, axes = mpf.plot(data,
                        type='candle', 
                        volume=True,
                        style=s, 
                        addplot=ap,
                        
                        figsize=(16, 10),
                        panel_ratios=(4,1,1),
                        returnfig=True)

    axes[0].set_title(f"{title}")
    axes[0].set_xticks(range(0, len(data), 30))  
    axes[0].set_xticklabels(range(0, len(data), 30))
    axes[4].set_ylabel('V_trades')

    for bar in axes[2].patches:
        bar.set_edgecolor('black')
        bar.set_linewidth(0.1)
    for bar in axes[4].patches:
        bar.set_edgecolor('black')
        bar.set_linewidth(0.1)

    return fig


def init_weights(model, args):

    d_model = args.emb_dim * len(args.voc_size)

    base_std = math.sqrt(2 / (5 * d_model))
    scaled_std = base_std / math.sqrt(2 * args.num_layers)

    for name, param in model.named_parameters():

        if 'make_emb' in name:
            nn.init.normal_(param, mean=0.0, std=base_std)
            param.data.mul_(math.sqrt(d_model))  # Scaled Embed

        elif ('att.1.output.weight' in name) or ('ff.3.weight' in name):
            nn.init.normal_(param, mean=0.0, std=scaled_std)

        elif 'bias' in name:
            nn.init.zeros_(param)

        else:
            nn.init.normal_(param, mean=0.0, std=base_std)



class MaskedMultiHeadAttention(nn.Module):

    def __init__(self, sequence, model_dim, dropout_prob, heads, rope=True):

        super().__init__()

        self.sequence = sequence
        self.model_dim = model_dim
        self.d_k = model_dim // heads
        self.heads = heads
        
       
        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.rope = rope
        self.scale = 1 / math.sqrt(self.d_k)
        self.register_buffer('mask', torch.tril(torch.ones(self.sequence, self.sequence)))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(model_dim, model_dim)

        if self.rope:

            base = 10000
            inv_freq = 1.0 / (base ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
            positions = torch.arange(sequence).float()
            angles = torch.einsum("i,j->ij", positions, inv_freq) 

            self.register_buffer('sin', torch.sin(angles))
            self.register_buffer('cos', torch.cos(angles))
        
        
    def apply_rotary_pos_emb(self, x, sin, cos):

        x1, x2 = x.view(*x.shape[:-1], self.d_k // 2, 2).unbind(-1)     

        x_rotated_1 = x1 * cos - x2 * sin
        x_rotated_2 = x1 * sin + x2 * cos
        x_rotated = torch.stack([x_rotated_1, x_rotated_2], dim=-1)  
        return x_rotated.view(x.shape) 

    def forward(self, x):

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = torch.stack(q.chunk(self.heads, dim=-1))
        k = torch.stack(k.chunk(self.heads, dim=-1))
        v = torch.stack(v.chunk(self.heads, dim=-1))

        if self.rope:
            q = self.apply_rotary_pos_emb(q, self.sin, self.cos)
            k = self.apply_rotary_pos_emb(k, self.sin, self.cos)


        scores = q @ k.transpose(-2,-1) / self.scale

        masked_scores = scores.masked_fill(self.mask == 0, float('-inf'))

        attn_weights = self.softmax(masked_scores)

        attn_weights = self.dropout(attn_weights)

        concat_heads = torch.cat([i for i in (attn_weights @ v)], dim=-1)
        
        return self.output(concat_heads)
    

class Block(nn.Module):
    def __init__(self, sequence, model_dim, ff_hidden_layer, dropout, num_heads):
        super().__init__()

        self.sequence = sequence
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ff_hidden_layer = ff_hidden_layer
        self.drop = dropout

        self.att = nn.Sequential(
            nn.LayerNorm(model_dim),
            MaskedMultiHeadAttention(sequence, model_dim, dropout, num_heads, rope=True),
            nn.Dropout(dropout)
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, ff_hidden_layer),
            nn.GELU(),
            nn.Linear(ff_hidden_layer, model_dim),
            nn.Dropout(dropout)
        )

    
    def forward(self, x):
        
        x = x + self.att(x)
        x = x + self.ff(x)

        return x


class GPT(nn.Module):
    def __init__(self, voc_size, sequence, emb_dim, ff_hidden_layer, dropout, num_heads, num_layers):
        super().__init__()

        self.sequence = sequence
        self.model_dim = emb_dim * len(voc_size)
        self.voc_size = voc_size
        self.num_heads = num_heads
        self.ff_hidden_layer = ff_hidden_layer
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.make_emb = nn.ModuleList([
            nn.Embedding(voc, emb_dim) 
            for voc in voc_size
            ])
        self.blocks = nn.Sequential(*[
            Block(sequence, self.model_dim, ff_hidden_layer, dropout, num_heads)
            for _ in range(num_layers)
        ])
        self.lin_out = nn.ModuleList([
            nn.Linear(self.model_dim, voc_size[i]) 
            for i in range(len(voc_size))
            ])
        self.log_soft = nn.LogSoftmax(dim=-1)


    def forward(self, x, y = None, pre_loss=False):

        x = torch.concatenate([self.make_emb[i](x[:,:,i]) for i in range(len(self.make_emb))], axis=-1)

        for block in self.blocks:
            x = block(x)

        x = [self.log_soft(W(x)) for W in self.lin_out]

        if pre_loss:

            return [torch.exp(i) for i in x]
        
        elif y == None:

            predict = torch.concatenate([torch.max(x[i], axis=-1).indices.unsqueeze(-1) for i in range(len(x))], axis=-1)

            return predict
        
        else:

            loss = torch.stack([
                nn.functional.nll_loss(x[i].reshape(-1,x[i].shape[-1]), y[:,:,i].reshape(-1)) 
                    for i in range(len(x))
                ]).mean()


            return loss



def train(args, start, model, optimizer, scheduler, train_loss, rank, world_size):

    # prepare model
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])

    work_done = 0
    if rank == 0:
        print(f'train log: {work_done}/100, time: {time.time() - start}, loss: {np.mean(train_loss[-100:])}', flush=True)

    data_loader = loader(args, train=True)

    # main train loop
    for x, y in data_loader:

        x = x.to(local_rank)
        y = y.to(local_rank)

        optimizer.zero_grad()

        loss = model(x, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_clip)

        optimizer.step()
        scheduler.step()

        loss_sum = loss.detach()
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        
        train_loss.append((loss_sum / world_size).item())
        progress = len(train_loss) % args.train_batches_per_epoch / args.train_batches_per_epoch * 100

        if work_done <= progress // 1 :

            work_done += 1

            if rank == 0:
                print(f'train log: {work_done}/100, time: {time.time() - start}, loss: {np.mean(train_loss[-100:])}', flush=True)


    return train_loss


def test(args, start, model, test_loss, rank, world_size):

    # prepare model
    model.eval()
    local_rank = int(os.environ['LOCAL_RANK'])

    loss_sum = 0
    if rank == 0:
        print(f'test log: 0/1, time: {time.time() - start}, loss: nan', flush=True)

    # main test loop
    with torch.no_grad():

        data_loader = loader(args, train=False)

        for x, y in data_loader:

            x = x.to(local_rank)
            y = y.to(local_rank)

            loss = model(x, y)

            loss_sum += loss.detach()

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        test_loss.append(loss_sum.item() / args.test_batches_per_epoch / world_size)

    if rank == 0:
        print(f'test log: 1/1 time: {time.time() - start}, loss: {test_loss[-1]}', flush=True)

    return test_loss
    

def final_test(args, start, model, exp_dir, rank, world_size):

    # prepare model and other variables for ploting metric for the final test on the best checkpoint
    model.eval()
    local_rank = int(os.environ['LOCAL_RANK'])

    test_dir = os.path.join(exp_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    num_features = len(args.voc_size)
    result = torch.zeros(args.sequence, num_features).to(local_rank)

    batches_passed = 0
    snapshots = np.linspace(1, args.test_batches_per_epoch, 10).astype(int)

    columns = ['open', 'high', 'low', 'close', 'volume_ema_50', 'volume_ema_500','v_trades_ema_50', 'v_trades_ema_500']
    start_data = pd.Series(np.ones(6), index=['open', 'high', 'low', 'close', 'volume', 'v_trades'])
    figures = []

    if rank == 0:
        print(f'final test log: 0/1, time: {time.time() - start}', flush=True)

    # main final test loop
    with torch.no_grad():

        data_loader = loader(args, train=False)

        for x, y in data_loader:

            x = x.to(local_rank)
            y = y.to(local_rank)

            pred = model(x)

            result += (pred == y).sum(axis=0)
            batches_passed += 1

            if (batches_passed in snapshots) and (rank == 0):

                rel_pred = vocab_to_rel(pred.cpu())
                rel_y = vocab_to_rel(y.cpu())

                abs_pred = relative_to_abs(pd.DataFrame(rel_pred[0], columns=columns), start_data)
                abs_y = relative_to_abs(pd.DataFrame(rel_y[0], columns=columns), start_data)

                figures.append(candle_plot(abs_pred, title='Predict'))
                figures.append(candle_plot(abs_y, title='True'))
            
            
            

        dist.all_reduce(result, op=dist.ReduceOp.SUM) 
        result = result.cpu()


    if rank == 0:
        
        # plot metrics
        pdf = PdfPages(os.path.join(test_dir, 'metrics.pdf'))

        fig = plt.figure(figsize=(16, 8), layout='constrained')
        gs = GridSpec(2, 4, figure=fig)
        axes = [fig.add_subplot(gs[i,o]) for i in range(2) for o in range(4)]

        for i in range(num_features):

            axes[i].plot(result[:,i] / args.test_batches_per_epoch / args.batch / world_size)
            axes[i].set_xlabel(f'var {i}')
        
        pdf.savefig(fig)

        for fig in figures:
            pdf.savefig(fig)

        pdf.close()
        

        print(f'final test log: 1/1, time: {time.time() - start}', flush=True)


def fsdp_main(model, args, exp_dir):
    start = time.time()

    # define devices
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    seed = 42 + local_rank
    np.random.seed(seed)
    torch.manual_seed(seed)

    if rank == 0:
        print(f'Total num. parameters: {args.total_parameters}', flush=True)
        print(f'Total num. GPU in use: {world_size}', flush=True)
        print(f'Total num. epochs: {args.epochs}', flush=True)

    # init processes
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    # shard model
    model = FSDP(model, device_id=torch.cuda.current_device())

    # establish optimizer
    optimizer = build_optimizer(model, 
                                args.lr, 
                                args.wd, 
                                args.corr_weight_decay)
    
    # establish scheduler
    total_steps = args.train_batches_per_epoch * args.epochs
    scheduler = get_linear_warmup_cosine_decay_scheduler(optimizer, 
                                                         total_steps,
                                                         warmup=args.warmup)

    # prepare additional variables
    best_test_loss = float('inf')
    checkp_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkp_dir, exist_ok=True)
    model_checkp = os.path.join(checkp_dir, 'model.pth')
    final_checkp = os.path.join(checkp_dir, 'model_final.pth')
    model_cpu_state = None
    final_model_cpu_state = None

    train_loss = []
    test_loss = []
    loss_png = os.path.join(exp_dir, 'loss_curve.png')


    
    # main loop
    for epoch in range(1, args.epochs + 1):

        # train
        dist.barrier()
        train_loss = train(args, start, model, optimizer, scheduler, train_loss, rank, world_size)

        # test
        dist.barrier()
        test_loss = test(args, start, model, test_loss, rank, world_size)

        # print results of an epoch
        if rank == 0:
            print(f"Epoch {epoch} finished, time: {time.time() - start}, train loss={np.mean(train_loss[-100:])}, test loss={test_loss[-1]}", flush=True)

            # update loss curve plot
            plt.figure()
            plt.semilogy(train_loss, label='train')
            plt.semilogy(np.linspace(1,len(train_loss),len(test_loss)), test_loss, label='test')
            plt.legend()
            plt.xlabel('Updates')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.savefig(loss_png)  
            plt.close()


        # save model if loss is less than was before
        if best_test_loss > test_loss[-1]:

            best_test_loss = test_loss[-1]
            
            dist.barrier()
            with FSDP.state_dict_type(
                model, 
                StateDictType.FULL_STATE_DICT, 
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                model_cpu_state = model.state_dict()


            if rank == 0:

                torch.save(model_cpu_state, model_checkp)
                print(f"Saved model, time {time.time() - start}", flush=True)


    # save final model
    dist.barrier()
    with FSDP.state_dict_type(
        model, 
        StateDictType.FULL_STATE_DICT, 
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        final_model_cpu_state = model.state_dict()


    if rank == 0:
        torch.save(final_model_cpu_state, final_checkp)
        print(f"Saved final model, time {time.time() - start}", flush=True)
        

    # load best model states
    dist.barrier()

    obj_list = [model_cpu_state]
    dist.broadcast_object_list(obj_list, src=0)
    model_cpu_state = obj_list[0]

    model.load_state_dict(model_cpu_state)


    # final test
    dist.barrier()
    final_test(args, start, model, exp_dir, rank, world_size)


    if rank == 0:
        print(f"finish", flush=True)


    dist.barrier()
    dist.destroy_process_group()







