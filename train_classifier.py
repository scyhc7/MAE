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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier-from_scratch.pt', help='best model (whole object); relative paths go under --output_dir')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--ckpt_name', type=str, default=None, help='resumable checkpoint under --output_dir; defaults to <output_model_path stem>-ckpt.pt')
    parser.add_argument('--resume', action='store_true', help='resume from the checkpoint if it exists')

    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    output_model_path = resolve_path(args.output_model_path, args.output_dir)
    ckpt_name = args.ckpt_name or os.path.splitext(os.path.basename(args.output_model_path))[0] + '-ckpt.pt'
    ckpt_path = resolve_path(ckpt_name, args.output_dir)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10(args.data_dir, train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10(args.data_dir, train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu', weights_only=False)
        log_name = 'pretrain-cls'
    else:
        model = MAE_ViT()
        log_name = 'scratch-cls'
    model = ViT_Classifier(model.encoder, num_classes=10).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    start_epoch = 0
    best_val_acc = 0
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
        best_val_acc = ckpt['best_val_acc']
        print(f'Resumed from {ckpt_path}: next epoch {start_epoch}, best_val_acc {best_val_acc}.')
    elif args.resume:
        print(f'No checkpoint at {ckpt_path}, starting from scratch.')

    writer = SummaryWriter(os.path.join(args.log_dir, 'cifar10', log_name), purge_step=start_epoch)

    for e in range(start_epoch, args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')       
            atomic_save(model, output_model_path)

        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
        writer.add_scalars('cls/acc', {'train' : avg_train_acc, 'val' : avg_val_acc}, global_step=e)

        writer.flush()
        atomic_save({
            'epoch': e,
            'step_count': step_count,
            'best_val_acc': best_val_acc,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'pending_grads': get_pending_grads(model),
            'rng': get_rng_state(),
            'args': vars(args),
        }, ckpt_path)

    writer.close()
