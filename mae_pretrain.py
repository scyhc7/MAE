import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed, resolve_path, atomic_save, get_rng_state, set_rng_state, \
    get_pending_grads, set_pending_grads, load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt', help='final model (whole object); relative paths go under --output_dir')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--ckpt_name', type=str, default='mae-ckpt.pt', help='resumable checkpoint written every epoch under --output_dir')
    parser.add_argument('--resume', action='store_true', help='resume from --output_dir/--ckpt_name if it exists')

    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = resolve_path(args.model_path, args.output_dir)
    ckpt_path = resolve_path(args.ckpt_name, args.output_dir)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10(args.data_dir, train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10(args.data_dir, train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    start_epoch = 0
    step_count = 0
    optim.zero_grad()
    if args.resume and os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optim'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        set_pending_grads(model, ckpt['pending_grads'])
        set_rng_state(ckpt['rng'])
        start_epoch = ckpt['epoch'] + 1
        step_count = ckpt['step_count']
        print(f'Resumed from {ckpt_path}: next epoch {start_epoch}, step_count {step_count}.')
    elif args.resume:
        print(f'No checkpoint at {ckpt_path}, starting from scratch.')

    # purge_step drops TensorBoard events logged after the checkpoint, so curves stay continuous on resume
    writer = SummaryWriter(os.path.join(args.log_dir, 'cifar10', 'mae-pretrain'), purge_step=start_epoch)

    for e in range(start_epoch, args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model (same format as before) and a resumable checkpoint '''
        writer.flush()
        atomic_save(model, model_path)
        atomic_save({
            'epoch': e,
            'step_count': step_count,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'pending_grads': get_pending_grads(model),
            'rng': get_rng_state(),
            'args': vars(args),
        }, ckpt_path)

    writer.close()
