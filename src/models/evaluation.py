import clip

import torch
from tqdm import tqdm
import logging
import numpy as np

from .. import datasets
from ..datasets.common import get_dataloader, maybe_dictionarize

logger = logging.getLogger(__name__)


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


# +
# @torch.no_grad()
# def zeroshot_classifier(classnames, templates, model):
#     if not isinstance(templates, list):
#         templates = [templates]
#     zeroshot_weights_prompt = []
#     zeroshot_weights_noprompt = []
#     for classname in classnames:
#         texts = [template(classname) for template in templates]  # format with class
#         texts = clip.tokenize(texts).cuda()  # tokenize
#         class_embeddings = model.encode_text(texts, use_prompt=True)[0]  # embed with text encoder
#         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#         class_embedding = class_embeddings.mean(dim=0)
#         class_embedding /= class_embedding.norm()
#         zeroshot_weights_prompt.append(class_embedding)
        
#         class_embeddings = model.encode_text(texts, use_prompt=False)[0]  # embed with text encoder
#         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#         class_embedding = class_embeddings.mean(dim=0)
#         class_embedding /= class_embedding.norm()
#         zeroshot_weights_noprompt.append(class_embedding)
        
#     zeroshot_weights_prompt = torch.stack(zeroshot_weights_prompt, dim=1).cuda()
#     zeroshot_weights_noprompt = torch.stack(zeroshot_weights_noprompt, dim=1).cuda()
#     return zeroshot_weights_prompt, zeroshot_weights_noprompt
# -

@torch.no_grad()
def zeroshot_classifier(classnames, templates, model):
    if not isinstance(templates, list):
        templates = [templates]
    
    all_zeroshot_weights = []
    for t in range(model.prompt_pool.task_count+2):
        zeroshot_weights = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            
            if t < model.prompt_pool.task_count+1:                
                _, _, text_prompt, text_ssf, _ = model.get_prompt(text=texts, taskid=t)
                class_embeddings = model.encode_text(texts, prompt=text_prompt, ssf=text_ssf)
            else:
                class_embeddings = model.encode_text(texts)
                
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
                
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
        all_zeroshot_weights.append(zeroshot_weights)

    return all_zeroshot_weights


# +
@torch.no_grad()
def zeroshot_eval(model, loader, zeroshot_weights, test_novelty=True, dataset_name=None):
    top1, top5, n = 0.0, 0.0, 0.0
    novel_count = 0
    
    all_probs = []
    for i, data in enumerate(loader):
#     for i, data in enumerate(tqdm(loader)):

        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        # predict
#         logits, novelty = model.zeroshot_inference(images, zeroshot_weights_prompt, zeroshot_weights_noprompt, test_novelty)
        task_id, novel = model.get_taskid(images)
        if not test_novelty:
            novel = False

#         all_probs.append(logits.softmax(dim=1).max(dim=1).values.cpu().numpy())
        
        if novel:
            logits = model(images, use_prompt=False, zeroshot_weights=zeroshot_weights[-1])[0]
            novel_count += len(images)
        else:
            logits = model(images, use_prompt=True, taskid=task_id, zeroshot_weights=zeroshot_weights[task_id])[0]
        
        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    logger.info(f"Total images: {n}")
    logger.info(f"Novel images: {novel_count}")
    
#     np.save(f'task{model.visual_prompt.task_count}_{dataset_name}_noguidedmix', np.concatenate(all_probs))
    return top1, top5


# +
# @torch.no_grad()
# def zeroshot_eval(model, loader, zeroshot_weights_prompt, zeroshot_weights_noprompt, test_novelty=True, dataset_name=None):
#     top1, top5, n = 0.0, 0.0, 0.0
#     novel_count = 0
    
#     all_probs = []
#     for i, data in enumerate(loader):
# #     for i, data in enumerate(tqdm(loader)):

#         data = maybe_dictionarize(data)
#         images = data["images"].cuda()
#         target = data["labels"].cuda()

#         # predict
# #         logits, novelty = model.zeroshot_inference(images, zeroshot_weights_prompt, zeroshot_weights_noprompt, test_novelty)
#         if test_novelty:
#             novelty = model.get_novelty(images)
#         else:
#             novelty = False

# #         all_probs.append(logits.softmax(dim=1).max(dim=1).values.cpu().numpy())
        
#         if novelty:
#             image_features = model.encode_image(images, use_prompt=False)[0]
#             zeroshot_weights = zeroshot_weights_noprompt
#             novel_count += len(images)
#         else:
#             image_features = model.encode_image(images, use_prompt=True)[0]
#             zeroshot_weights = zeroshot_weights_prompt
        
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         logits = 100.0 * image_features @ zeroshot_weights

#         # measure accuracy
#         acc1, acc5 = accuracy(logits, target, topk=(1, 5))
#         top1 += acc1
#         top5 += acc5
#         n += images.size(0)

#     top1 = (top1 / n) * 100
#     top5 = (top5 / n) * 100
#     logger.info(f"Total images: {n}")
#     logger.info(f"Novel images: {novel_count}")
    
# #     np.save(f'task{model.visual_prompt.task_count}_{dataset_name}_noguidedmix', np.concatenate(all_probs))
#     return top1, top5

# +
def eval_single_dataset(image_classifier, dataset, args, test_novelty=True, dataset_name=None):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()

#     model.visual_prompt.count[:] = 0    
#     model.text_prompt.count[:] = 0    
    
    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model
    )
#     zeroshot_weights_prompt, zeroshot_weights_noprompt = zeroshot_classifier(
#         dataset.classnames, dataset.templates, model
#     )

    model.prompt_pool.count[:] = 0    

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights, test_novelty, dataset_name)
#     top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights_prompt, zeroshot_weights_noprompt, test_novelty, dataset_name)

    if test_novelty:
        logger.info(f"Key count: {model.prompt_pool.count}")
#         logger.info(f"Key count (Text): {model.text_prompt.count}")
#         logger.info(f"Key count (Visual): {model.visual_prompt.count}")
    
    logger.info(f"Top-1 accuracy: {top1:.2f}")
    # print(f"Top-5 accuracy: {top5:.2f}")


# -

def evaluate(image_classifier, args, val_preprocess, task_id=None, test_novelty=True):
    if args.eval_datasets is None:
        return
    for i, dataset_name in enumerate(args.eval_datasets):
        if task_id is not None and i not in task_id:
            continue
            
        logger.info(f"Evaluating on {dataset_name}")
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        eval_single_dataset(image_classifier, dataset, args, test_novelty, dataset_name)
