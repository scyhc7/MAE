import os
import random
import torch
import numpy as np

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def resolve_path(path, base_dir):
    ''' Relative paths are placed under base_dir; absolute paths are kept as is. '''
    return path if os.path.isabs(path) else os.path.join(base_dir, path)

def atomic_save(obj, path):
    ''' Write to a temp file then rename, so an interrupted write never corrupts an existing file. '''
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def get_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])

def get_pending_grads(model):
    ''' Gradients accumulated but not yet applied (gradient accumulation can span epochs). '''
    return [None if p.grad is None else p.grad.detach().cpu().clone() for p in model.parameters()]

def set_pending_grads(model, grads):
    for p, g in zip(model.parameters(), grads):
        p.grad = None if g is None else g.to(p.device)

def load_checkpoint(path):
    # The checkpoint is produced by this project and contains RNG states (numpy arrays, tuples),
    # so weights_only must be False. Only load checkpoints you created yourself.
    return torch.load(path, map_location='cpu', weights_only=False)
