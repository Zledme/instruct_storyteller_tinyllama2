#!/usr/bin/env python
# coding: utf-8

# # imports

# In[70]:


import sys
sys.path.append('../src/')
import math
import os
import glob
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from contextlib import nullcontext
import matplotlib.pyplot as plt
from sentencepiece import SentencePieceProcessor
from data_loader import *
from utils import *
from model import *


# In[ ]:





# In[71]:


device = 'cuda:0'
device_type = 'cuda'


# # paths

# In[72]:


DATA_CACHE_DIR = '../instruct_dataset/'


# In[73]:


out_dir = '../models/'
os.makedirs(out_dir, exist_ok=True)


# # tokenizer

# In[74]:


tokenizer = SentencePieceProcessor('../tokenizer.model')


# In[75]:


vocab_size = tokenizer.vocab_size()


# # training

# #### mixed precision settings

# In[76]:


dtype = 'bfloat16'
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu" 
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


# In[77]:


scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))


# #### model

# In[78]:


dim = 288 #288 #
n_layers =  20
n_heads =  8
n_kv_heads = n_heads
multiple_of = 32
dropout = 0.0
max_seq_len = 350


# In[79]:


model_args = ModelArgs(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_heads,
    vocab_size=32000,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
) 


# In[80]:


model = Transformer(model_args)
model.to(device);
print(f'Number of parameters: {sum(p.nelement() for p in model.parameters())}')


# #### data

# In[81]:


batch_size = 32

wanted_batch_size = 4 * 128
# gradient_accumulation_steps = wanted_batch_size // batch_size
gradient_accumulation_steps = 1
print(f'Wanted batch_size: {wanted_batch_size}, gradient accumulation steps: {gradient_accumulation_steps}, batch_size: {batch_size}')


# In[82]:


iter_batches = partial(
    iter_batch_func,
    device=device,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    data_cache_dir=DATA_CACHE_DIR
)


# #### optimizer

# In[83]:


learning_rate = 5e-4
optimizer = get_optimizer(
    model=model,
    device_type='cuda',
    learning_rate=learning_rate,  # max learning rate
    weight_decay = 1e-1,
    beta1 = 0.9,
    beta2 = 0.95,
)


# ## training loop

# In[84]:


max_iters = 50000
eval_iters = 100
best_val_loss = 1e9
grad_clip = 1


# In[85]:


eval_prompt = 'Write a story. In the story, try to use the verb "eat", the noun "clock" and the adjective "clever". The story has the following features: the story should contain at least one dialogue. Possible story:'


# In[86]:


import time
warmup_iters = 1000
eval_interval = 2000
eval_only = False
always_save_checkpoint = True
log_interval= 1
iter_num = 0
lr_decay_iters = max_iters
min_lr = 0.0


# In[ ]:





# In[87]:


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < 1000:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
# training loop
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model  # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if True else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 :
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # if wandb_log:
        #     try:
        #         wandb.log(
        #             {
        #                 "iter": iter_num,
        #                 "tokens": iter_num * tokens_per_iter,
        #                 "loss/train": losses["train"],
        #                 "loss/val": losses["val"],
        #                 "lr": lr,
        #                 "mfu": running_mfu * 100,  # convert to percentage
        #             }, step = iter_num
        #         )
        #     except Exception as e:
        #         print(f"logging to wandb failed: {e}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                #model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        #if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
        #    model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 :
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break


# In[ ]:





# In[ ]:




