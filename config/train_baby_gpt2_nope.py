
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

out_dir = 'out-baby-gpt'

wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-baby'

# these make the total batch size be ~0.5M
# 64 batch size * 1024 block size * 4 gradaccum  = 491,520
batch_size = 64
block_size = 1024
gradient_accumulation_steps = 8

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
# dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially


# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

pe = 'nope'
flash = True
