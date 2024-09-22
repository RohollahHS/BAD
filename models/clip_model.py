import my_clip
import torch
import os


def load_clip_model(args):
    clip_model = my_clip.load("ViT-B/32", device=args.device, jit=False,
                              download_root=os.path.join(os.environ.get("TORCH_HOME"), 'clip'))  # Must set jit=False for training
    my_clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model

