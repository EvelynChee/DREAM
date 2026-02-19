import copy
import os
from random import random

import clip
import torch

from . import utils
from .args import parse_arguments
from .models import evaluate, finetune_ours, finetune_ours_semi
from .models.modeling import create_image_classifier


def merge(model_0, model_1, alpha=0.95):
    key_name = [k for k, v in model_0.named_parameters()]
    for i, (param_q, param_k) in enumerate(zip(model_0.parameters(), model_1.parameters())):
        param_k.data = param_k.data * alpha + param_q.data * (1 - alpha)
    return model_1


def main(args):
    print(args)
    utils.seed_all(args.seed)

    if args.eval_only:
        model, _, val_preprocess = clip.load(args.model, jit=False)
        if args.load:
            utils.torch_load(model, args.load)
        elif args.save:
            checkpoint_pth = os.path.join(
                args.save, f"clip_zeroshot_{args.train_dataset}.pth"
            )
            utils.torch_save(checkpoint_pth, model)
        evaluate(model, args, val_preprocess)
    else:
        if args.method in ['ours_semi']:
            model = finetune_ours_semi(args)
        else:
            model = finetune_ours(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
