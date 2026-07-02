# Copyright (c) Open-MMLab. All rights reserved.
# Patched to remove hard mmcv dependency — works with any mmcv version or none.
import os
import os.path as osp
import warnings
from collections import OrderedDict

import torch
from torch.nn import functional as F

try:
    from mmcv.parallel import is_module_wrapper as _mmcv_is_module_wrapper
    from mmcv.runner import get_dist_info as _mmcv_get_dist_info
    _HAS_MMCV = True
except Exception:
    _HAS_MMCV = False


def is_module_wrapper(module):
    if _HAS_MMCV:
        try:
            return _mmcv_is_module_wrapper(module)
        except Exception:
            pass
    return hasattr(module, 'module') and isinstance(module.module, torch.nn.Module)


def get_dist_info():
    if _HAS_MMCV:
        try:
            return _mmcv_get_dist_info()
        except Exception:
            pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def mkdir_or_exist(dir_name):
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [k for k in all_missing_keys if 'num_batches_tracked' not in k]

    if unexpected_keys:
        err_msg.append(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg_str = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg_str)
        elif logger is not None:
            logger.warning(err_msg_str)
        else:
            print(err_msg_str)


def _load_checkpoint(filename, map_location=None):
    if filename.startswith(('http://', 'https://')):
        checkpoint = torch.hub.load_state_dict_from_url(filename, map_location=map_location)
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v
                      for k, v in state_dict.items() if k.startswith('encoder.')}

    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 == N2 and C1 == C2 and L == H * W:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)

    relative_position_bias_table_keys = [k for k in state_dict.keys()
                                         if 'relative_position_bias_table' in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 == nH2 and L1 != L2:
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            table_pretrained_resized = F.interpolate(
                table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                size=(S2, S2), mode='bicubic')
            state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)

    load_state_dict(model, state_dict, strict, logger)
    return checkpoint
