# -*- coding: utf-8 -*-
"""
Varianta minimala, fara dependenta de mmcv, pentru HuggingFace Space.
Incarca checkpoint-ul pre-trained TinyViT direct cu torch (suficient,
pentru ca greutatile sunt suprascrise imediat de encoder.pth-ul propriu).
"""
import torch


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


__all__ = ['load_checkpoint']
