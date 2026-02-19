import collections
import logging
import copy
import os
import numpy as np
import clip.clip as clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from . import gradcam
from .. import datasets, templates, utils
from .evaluation import evaluate, zeroshot_classifier
from .helpers import get_datasets_text, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation

logging.basicConfig(
    format='%(asctime)s [%(filename)s]: %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO
)
logger = logging.getLogger(__name__)


# +
def ours(args):
    assert args.dataset_order is not None, "order need to be provided"
    assert args.train_dataset is None

    dataset_order = args.dataset_order
    args.eval_datasets = dataset_order

    lrs = args.lr
    if len(lrs) < len(dataset_order):
        lrs = [lrs[0]] * len(dataset_order)
        
    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, args=args)  # model='ViT-B/16'

    base_model = copy.deepcopy(model)
    base_model.prompt_pool = None
#     base_model.visual_projection_new = None
#     base_model.text_projection_new = None
#     base_model.text_prompt = None
#     base_model.visual_prompt = None
    base_model.eval()
        
#     evaluate(prev_model, args, val_preprocess)
    
    
    for i in np.arange(len(dataset_order)):
        train_dataset = dataset_order[i]
        args.train_dataset = train_dataset
        args.lr = lrs[i]
        logger.info(f'[Dataset] {train_dataset}')
        logger.info(f'[LR] {args.lr}')
                        
        if i > 0:
            model.prompt_pool.process_task_count()
#             model.visual_prompt.process_task_count()
#             model.text_prompt.process_task_count()
            
        model = finetune(args, model, base_model, train_preprocess, val_preprocess, i)
        
#         args.load = os.path.join(args.save, f"{dataset_order[i]}.pth")
        
        evaluate(model, args, val_preprocess)
        
        base_model = copy.deepcopy(model)
        base_model.eval()
        
#         if i == 5:
#             evaluate(model, args, val_preprocess)
#             break

# +
class ImageClassifier(nn.Module):
    def __init__(self, model, text_tokens, check_novelty=False):
        super(ImageClassifier, self).__init__()
        self.model = model
        self.text_tokens = text_tokens
        self.check_novelty = check_novelty
#         self.text_features = text_features

#         self.text_features = model.encode_text(text_tokens)[0]  # embed with text encoder
#         self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)            
    def forward(self, x):
        logits_per_image = self.model(x, self.text_tokens, taskid=-1, is_train=False, use_prompt=True, check_novelty=self.check_novelty)[0]
#         image_features = self.model.encode_image(x, task_id=-1, train=False)[0]
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         logits_per_image = self.model.logit_scale.exp() * image_features @ self.text_features.t()
        
        probs = logits_per_image.softmax(dim=1)

        return probs
    
    
def reshape_transform(tensor, height=14, width=14):
    tensor = tensor.transpose(0,1)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def set_novelty_threshold(model, val_preprocess, task_id, text_tokens, dataset, save_path, alpha, zeta, load_path=None, flip=False):
    logger.info('Computing threshold')
    
#     if dataset.name in ['caltech101', 'cifar100']:
#         alpha, zeta = 10, 1
        
# #         zeta = 0.1
# #     print(alpha, zeta)
#     if task_id <= 1:
        
# # #         prev = [0.7181631, 0.57, 0.5071908, 0.63, 0.6340534, 0.7475095, 0.6224972, 0.52866334, 0.7084005, 0.6229546, 0.5472152]
# # # #         prev = [0.7181631, 0.53627, 0.5071908, 0.69251233, 0.6340534, 0.7475095, 0.6224972, 0.52866334, 0.7084005, 0.6229546, 0.5472152]
        
# #         prev = [0.018056586, 0.0195, 0.019, 0.020566259, 0.019372039, 0.017452475, 0.020439092, 0.022708558, 0.019423079, 0.01983711, 0.022090228]
# # #         prev = [0.018056586, 0.021604592, 0.020805895, 0.020566259, 0.019372039, 0.017452475, 0.020439092, 0.022708558, 0.019423079, 0.01983711, 0.022090228]
        
#         prev = [0.6262131, 0.6112303]
        
#         threshold =  prev[task_id]
#         model.prompt_pool.set_novelty_threshold(threshold, task_id)
#         logger.info(model.prompt_pool.nov_threshold)
#         return threshold

    if load_path is not None and os.path.exists(f'{load_path}/novel_sim_t{task_id}.npy'):
        all_mix_sim, all_ori_sim, weights = np.load(f'{load_path}/novel_sim_t{task_id}.npy')
    else:
        model.eval()
        data_loader = dataset.train_loader
        
#         temp_dataset = dataset.train_dataset
#         if isinstance(temp_dataset, torch.utils.data.dataset.Subset):       
#             if isinstance(temp_dataset.dataset, torch.utils.data.dataset.Subset):       
#                 temp_dataset.dataset.dataset.transform = val_preprocess
#                 temp_dataset.dataset.dataset.gradcam = None
#             else:
#                 temp_dataset.dataset.transform = val_preprocess
#                 temp_dataset.dataset.gradcam = None
#         else:
#             temp_dataset.transform = val_preprocess
#             temp_dataset.gradcam = None

#         data_loader = torch.utils.data.DataLoader(
#             temp_dataset,
#             batch_size=dataset.batch_size,
#             shuffle=True,
#             num_workers=dataset.num_workers,
#         )
        current_model = ImageClassifier(model, text_tokens).cuda()
        current_model.eval()
        target_layers = [current_model.model.visual.transformer.resblocks[-1].ln_1]

        cam = gradcam.GradCAM(model=current_model, target_layers=target_layers, reshape_transform=reshape_transform)

    #     ori_novelpred = [] 
    #     mix_novelpred = [] 
        weights = []

        all_ori_sim = []
        all_mix_sim = []
    #     all_ori_logit = []
    #     all_mix_logit = []
        for batch in data_loader:
            inputs, targets = batch[0], batch[1]
            inputs, targets = inputs.cuda(), targets.cuda()

            gcam = cam(input_tensor=inputs, targets=targets)
            mix_inputs, mask, _ = guidedmix(gcam, inputs, targets, alpha, zeta, random=True)
            mix_inputs = mix_inputs #+ torch.randn_like(mix_inputs) * 0.1
            lam = (1-mask).view(inputs.size()[0], -1).sum(dim=1, keepdim=True) / (inputs.size()[-1] * inputs.size()[-2])

            with torch.no_grad():

    #             mix_sim, mix_logit = model.get_sim_and_logit(mix_inputs, text_features.T)
    #             ori_sim, ori_logit = model.get_sim_and_logit(inputs, text_features.T)

    #             all_mix_sim.append(mix_sim.cpu().numpy())
    #             all_mix_logit.append(mix_logit.max(dim=1).values.cpu().numpy())
    #             all_ori_sim.append(ori_sim.cpu().numpy())
    #             all_ori_logit.append(ori_logit.max(dim=1).values.cpu().numpy())

                all_mix_sim.append(model.get_sim(mix_inputs).cpu().numpy())
                all_ori_sim.append(model.get_sim(inputs).cpu().numpy())            

                weights.append(lam.cpu().numpy())

                if flip:
                    all_mix_sim.append(model.get_sim(torch.flip(mix_inputs, dims=[3])).cpu().numpy())
                    all_ori_sim.append(model.get_sim(torch.flip(inputs, dims=[3])).cpu().numpy())            
                    weights.append(lam.cpu().numpy())        
                
    #             mix_novelpred.append(model.get_sim(mix_inputs).cpu().numpy())
    #             ori_novelpred.append(model.get_sim(image=inputs).cpu().numpy())            
    #             weights.append(lam.cpu().numpy())

    #     np.save(f'sample_image_{task_id}.npy', [inputs.cpu().numpy(), mix_inputs.cpu().numpy()])

    #     mix_novelpred = np.concatenate(mix_novelpred)
    #     ori_novelpred = np.concatenate(ori_novelpred)

        all_mix_sim = np.concatenate(all_mix_sim)
        all_ori_sim = np.concatenate(all_ori_sim)
    #     all_mix_logit = np.concatenate(all_mix_logit)
    #     all_ori_logit = np.concatenate(all_ori_logit)

        weights = np.concatenate(weights)[:,0]
                    
        del current_model
        del cam
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)        
        np.save(f'{save_path}/novel_sim_t{task_id}.npy', np.stack([all_mix_sim, all_ori_sim, weights]))
#     np.save(f'novel_logit_t{task_id}.npy', np.stack([all_mix_logit, all_ori_logit, weights]))
    
    weights = np.minimum(1-weights, weights) 

    threshold_sim = np.sum(weights * all_mix_sim)/np.sum(weights)
#     threshold_sim = 0.55
#     threshold_logit = np.sum(weights * all_mix_logit)/np.sum(weights)
#     threshold_sim = np.percentile(all_ori_sim, 95)
#     threshold_sim = 1-np.percentile(1-all_ori_sim, 85)
#     threshold_logit = 1-np.percentile(1-all_ori_logit, 99)
    
#     prev = [0.7181631, 0.57, 0.5071908, 0.63, 0.6340534, 0.7475095, 0.6224972, 0.52866334, 0.7084005, 0.6229546, 0.5472152]
#     threshold_sim = prev[task_id]
    
    model.prompt_pool.set_novelty_threshold(threshold_sim, task_id)
    logger.info(f'Novelty threshold: {model.prompt_pool.nov_threshold}')
#     model.visual_prompt.set_novelty_threshold(threshold_sim, task_id)
#     logger.info(f'Visual threshold (sim): {model.visual_prompt.sim_threshold}')

#     model.set_novelty_threshold(threshold_logit, task_id)
#     logger.info(f'Visual threshold (logit): {model.logit_threshold}')


#     with torch.no_grad():
#         ori_novelpred = model.get_sim(text=text_tokens).cpu().numpy()
#     threshold = 1-np.percentile(1-ori_novelpred, 99)
#     model.text_prompt.set_novelty_threshold(threshold, task_id)
#     logger.info(f'Text threshold: {model.text_prompt.sim_threshold}')
    
    return

def guidedmix(gcam, image, targets, alpha, zeta, random=True):
    batch_size, _, H, W = image.size()
#     mask = ~(targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t()))

    if not isinstance(gcam, torch.Tensor):
        gcam = torch.tensor(gcam).to(image.device).unsqueeze(1)
#         gcam = torch.tensor(gcam).cuda().unsqueeze(1)
        
    sim = gcam.view(batch_size, -1) @ gcam.view(batch_size, -1).T
    sim = sim / (H*W)
    
    if random:
#         sim = ~(targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())) 
        sim = sim + torch.rand_like(sim) * 0.1
    
#     mask = mask * 10 + sim
    sim.fill_diagonal_(0)
        
#     sim = mask * sim
#     if cls_emb is not None:
#         sim_cls = cls_emb[targets] @ cls_emb[targets].t()
#         sim = sim * sim_cls
    
    index = torch.argsort(sim, dim=-1, descending=True)[:,0]
    
    mask = torch.sigmoid(alpha * (gcam.clamp(0,1)**zeta - 0.5))
    mix_image = (1-mask) * image + mask * image[index]
    return mix_image, mask, index    


# +
def generate_gradcam(dataset, val_preprocess, model, text_tokens, args):
    logger.info('Generating GradCAM')
#     dataset = dataset_class(
#         val_preprocess,
#         location=args.data_location,
#         batch_size=args.batch_size,
#     )
        
    
    temp_dataset = dataset.train_dataset
    if isinstance(temp_dataset, torch.utils.data.dataset.Subset):       
        if isinstance(temp_dataset.dataset, torch.utils.data.dataset.Subset):       
            original_preprocess = temp_dataset.dataset.dataset.transform
            temp_dataset.dataset.dataset.transform = val_preprocess
        else:
            original_preprocess = temp_dataset.dataset.transform
            temp_dataset.dataset.transform = val_preprocess
    else:
        original_preprocess = temp_dataset.transform
        temp_dataset.transform = val_preprocess
    
    data_loader = torch.utils.data.DataLoader(
        temp_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )    
    
#     if os.path.exists(f'dataset_gradcam_backup/{dataset.name}.npy'):
#         all_gcam = np.load(f'dataset_gradcam_backup/{dataset.name}.npy')

#     else:
    current_model = ImageClassifier(model, text_tokens, check_novelty=True).cuda()
    current_model.eval()
    for p in current_model.parameters():
        p.requires_grad = True

    target_layers = [current_model.model.visual.transformer.resblocks[-1].ln_1]

    cam = gradcam.GradCAM(model=current_model, target_layers=target_layers, reshape_transform=reshape_transform)

    all_gcam = []
    for inputs, targets in data_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        gcam = cam(input_tensor=inputs, targets=targets, return_target_size=False)
        all_gcam.append(gcam)

    all_gcam = np.concatenate(all_gcam)
    del current_model
    del cam

#     np.save(f'dataset_gradcam/{dataset.name}_base.npy', all_gcam)

    if isinstance(temp_dataset, torch.utils.data.dataset.Subset):       
        if isinstance(temp_dataset.dataset, torch.utils.data.dataset.Subset):       
            dataset.train_dataset.dataset.dataset.transform = original_preprocess
        else:
            dataset.train_dataset.dataset.transform = original_preprocess
    else:
        dataset.train_dataset.transform = original_preprocess
    dataset.set_gradcam(all_gcam)
    
    
    return 


# +
def dataset_selection(model, dataloader, texts, k=5):
    few_shot_data = {} # create few_shot data

    labels = []    
    preds = []
    
    count = 0
    with torch.no_grad():
        for images, targets in dataloader:        
            images = images.cuda()
            
            probs = model.get_sim(images, last=False)
            preds.append(probs.min(dim=1)[0].cpu().numpy())
            
#             probs = F.softmax(model(images, texts, taskid=-1, is_train=False, check_novelty=True)[0], dim=1)
#             preds.append(probs.max(dim=1)[0].cpu().numpy())
            labels.append(targets.numpy())
    
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    for cls in np.unique(labels):
        cls_idx = np.where(labels==cls)[0]
        cls_preds = preds[cls_idx]
#         few_shot_data[cls] = cls_idx[np.argsort(cls_preds)[:k]]
        few_shot_data[cls] = cls_idx[np.argsort(cls_preds)[::(len(cls_idx)-1)//(k-1)]]
        
    few_shot_data_idx =  np.concatenate([v for k,v in few_shot_data.items()])

    assert len(few_shot_data_idx) == k * len(np.unique(labels))
    
    return few_shot_data_idx


# +
def finetune(args, model, base_model, train_preprocess, val_preprocess, task_id):        
    # prepare dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )

    # prepare template
    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    # text
    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()
    
    mix_texts = [template(f'mix of {c1} and {c2}' if c1 != c2 else f'mix of {c1}') for c1 in dataset.classnames for c2 in dataset.classnames]
    mix_texts = clip.tokenize(mix_texts).cuda()

    alpha, zeta = 5, 0.5
    if args.train_dataset in ['Caltech101']:
        alpha, zeta = 10.0, 1.0
#     elif args.train_dataset in ['DTD']:
#         alpha, zeta = 1.0, 0.1
        
#     if task_id > 0:
#         check_novelty(model, task_id, texts, dataset)

    if args.few_shot > 0:
        logger.info('=====few-shot======')
        # few-shot
#         few_shot_data = {} # create few_shot data

#         for idx, (image, label) in enumerate(dataset.train_loader_noshuffle):
#             label = label.item()
#             if label not in few_shot_data:
#                 few_shot_data[label] = []
#             if len(few_shot_data[label]) < args.few_shot:
#                 few_shot_data[label].append(idx)
#             else:
#                 j = np.random.randint(0, idx+2)
#                 if j < args.few_shot:
#                     few_shot_data[label][j] = idx
        
#         few_shot_data_idx = sum([v for k, v in few_shot_data.items()], [])
        
        few_shot_data_idx = dataset_selection(base_model, dataset.train_loader_noshuffle, texts, k=args.few_shot)            
        
#         few_shot_data_idx = np.load(f'few_shot_{args.train_dataset}.npy')
        dataset.update_train_loader(few_shot_data_idx)

#         np.save(f'few_shot_{args.train_dataset}', few_shot_data_idx)
        
#     dataset_gradcam = generate_gradcam(dataset_class, val_preprocess, base_model, texts, args)
#     if args.few_shot > 0:
#         dataset_gradcam = dataset_gradcam[few_shot_data_idx]
#     dataset.set_gradcam(dataset_gradcam)
    
#     if task_id > 4:
#         args.load = None
        
    if args.load is not None and os.path.exists(os.path.join(args.load, f"{args.train_dataset}.pth")):
        utils.torch_load(model, os.path.join(args.load, f"{args.train_dataset}.pth"))
        
#         freeze_prompt(model, dataset, texts)
        set_novelty_threshold(model, val_preprocess, task_id, texts, dataset, args.save, alpha, zeta, None)
    
        return model

    data_iter = iter(dataset.train_loader)
    num_batches = len(dataset.train_loader)

    generate_gradcam(dataset, val_preprocess, base_model, texts, args)
        
    if args.epochs is not None: # # False
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations  # 1000
    if args.eval_every_epoch:  # False
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval # none
    loss_interval = args.loss_interval
    logger.info(f"Iterations per epoch: {num_batches}")
    logger.info(f"Total iterations: {total_iterations}")

    # get params
    logger.info("[Training mode] Prompts")
    for k, v in model.named_parameters():  # forzen params
        if "prompt" not in k:
            v.requires_grad = False

    params = [
        v for k, v in model.named_parameters() if "prompt" in k and v.requires_grad
    ]
    params_name = [
        k for k, v in model.named_parameters() if "prompt" in k and v.requires_grad
    ]
    logger.info(f'trainable params: {params_name}')

    # print trainable params's information
    total_params_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    logger.info(f'The number of Total Trainable Parameters------------------: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    logger.info(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")

    # optimizer
#     params = [
#         {
#             'params': [v for k, v in model.named_parameters() if 'prompt' in k and v.requires_grad and (not ('key' in k) and not ('weights' in k))],
#             'lr': args.lr
#         },
#         {
#             'params': [v for k, v in model.named_parameters() if 'prompt' in k and v.requires_grad and ('key' in k or 'weights' in k)],
#             'lr': 0.008
#         },
#     ]
        
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations, args.min_lr
    )
    
    # move model to device
    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    logger.info(f"Using devices {devices}")
    model = torch.nn.DataParallel(model, device_ids=devices) # 模型并行化

    metrics = collections.defaultdict(float)
    
    for iteration in range(total_iterations):
#     for iteration in tqdm(range(total_iterations)):
        if eval_iterations is not None and (iteration + 1) % eval_iterations == 0:
            evaluate(model.module, args, val_preprocess, np.arange(task_id+1), test_novelty=False)

        # prepare model
        model.train()
        scheduler(iteration)

        # prepare data
        try:
            images, labels, gradcam = next(data_iter)
        except:
            data_iter = iter(dataset.train_loader)
#             data_iter = iter(few_shot_data_loader)
            images, labels, gradcam = next(data_iter)
        images, labels, gradcam = images.cuda(), labels.cuda(), gradcam.cuda()
        bs = images.shape[0]

        mix_images, mask, mix_idx = guidedmix(gradcam, images, labels, alpha, zeta)
        lam = (1-mask).view(images.size()[0], -1).sum(dim=1, keepdim=True) / (images.size()[-1] * images.size()[-2])
        lam = lam.squeeze(dim=1)
        mix_targets = len(dataset.classnames) * labels + labels[mix_idx]
        mix_texts_sub = mix_texts[mix_targets.unique()]
        mix_targets = (mix_targets.unsqueeze(dim=1) == mix_targets.unique().unsqueeze(dim=0)).nonzero()[:,1]
        
#         images = torch.cat([images, mix_images])
#         images = torch.cat([images[:bs//2], mix_images[bs//2:]])

        logits_per_image, _, prompt_loss, image_feature, text_feature = model(images, texts, task_id, is_train=True)  # 分开        
        mix_logits_per_image, _, mix_prompt_loss, _, _ = model(mix_images, mix_texts_sub, task_id, is_train=True)  # 分开        
        
        # -- cross entropy loss --
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)
#         loss = F.cross_entropy(logits_per_image[:bs//2], labels[:bs//2], label_smoothing=args.ls)
#         loss = F.cross_entropy(logits_per_image[:bs], labels, label_smoothing=args.ls)
        metrics["clf_loss"] += loss.item()          

#         prompt_loss = text_prompt_loss + image_prompt_loss
#         prompt_loss = image_prompt_loss.mean() + text_prompt_loss.mean()
#         prompt_loss = prompt_loss[:bs//2].mean() + (prompt_loss[:bs//2].mean() - prompt_loss[bs//2:].mean()) 
        prompt_loss = prompt_loss.mean() + (prompt_loss.mean() - mix_prompt_loss.mean()) 
#         prompt_loss = prompt_loss[:bs].mean() + (prompt_loss[:bs].mean() - prompt_loss[bs:].mean()) 
#         prompt_loss = prompt_loss[:bs].mean() + (lam * (prompt_loss[:bs] - prompt_loss[bs:])).mean() 
#         prompt_loss = prompt_loss.mean()
        loss += prompt_loss
        metrics["aux_loss"] += prompt_loss.item()   

#         mix_image_prompt_loss = model.module.get_prompt_loss(mix_images, task_id)
#         mix_prompt_loss = (torch.minimum(lam,1-lam) * mix_image_prompt_loss).mean()
#         loss += mix_prompt_loss
#         metrics["mixaux_loss"] += mix_prompt_loss.item()   
        
# #         mix_logits_per_image = model(mix_images[bs//2:], texts, task_id, is_train=True)[0]  # 分开        
#         mix_loss = (
#             lam[bs//2:] * F.cross_entropy(logits_per_image[bs//2:], labels[bs//2:], reduction='none') + \
#             (1-lam[bs//2:]) * F.cross_entropy(logits_per_image[bs//2:], labels[mix_idx][bs//2:], reduction='none')
#         ).mean() * 0.1
#         mix_loss = (
#             lam * F.cross_entropy(logits_per_image[bs:], labels, reduction='none') + \
#             (1-lam) * F.cross_entropy(logits_per_image[bs:], labels[mix_idx], reduction='none')
#         ).mean() * 0.1
        mix_loss = ((torch.minimum(lam, 1-lam)**2) * F.cross_entropy(mix_logits_per_image, mix_targets, reduction='none', label_smoothing=args.ls)).mean() #* 0.1  
        loss += mix_loss        
        metrics["mixclf_loss"] += mix_loss.item()       
                
#         with torch.no_grad():
#             prev_logits_per_image , _, _, prev_image_feature, prev_text_feature = base_model(images, texts, -1, is_train=False, check_novelty=True)
        
#         T = 2
#         prev_conf = F.softmax(prev_logits_per_image, dim=1).max(dim=1).values
#         dis_loss = (F.kl_div( \
#                     F.log_softmax(logits_per_image/T, dim=1), \
#                     F.softmax(prev_logits_per_image.detach()/T, dim=1), reduction='none').sum(dim=1) * \
#                     prev_conf).mean() * T * T 

# #         sim_image = logit_scale.exp() * prev_image_feature.detach() @ image_feature.T
# #         sim_text = logit_scale.exp() * prev_text_feature.detach() @ text_feature.T

# #         sim_loss = 0.5 * (
# #             F.cross_entropy(sim_image, torch.arange(len(labels)).cuda()) + \
# #             F.cross_entropy(sim_image.T, torch.arange(len(labels)).cuda())
# #         ) + 0.5 * (
# #             F.cross_entropy(sim_text, torch.arange(len(texts)).cuda()) + \
# #             F.cross_entropy(sim_text.T, torch.arange(len(texts)).cuda()) 
# #         )
#         loss += dis_loss        
#         metrics["dis_loss"] += dis_loss.item()       

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#         metrics["acc"] += (labels == logits_per_image.argmax(axis=1)).float().mean().item()
#         metrics["acc"] += (labels[:bs//2] == logits_per_image[:bs//2].argmax(axis=1)).float().mean().item()
        metrics["acc"] += (labels == logits_per_image[:bs].argmax(axis=1)).float().mean().item()

        # evaluation
        if (iteration + 1) % num_batches == 0:
            print_metrics(metrics, num_batches, (iteration + 1) // num_batches)
            metrics = collections.defaultdict(float)

#     freeze_prompt(model.module, dataset, texts)
    set_novelty_threshold(model.module, val_preprocess, task_id, texts, dataset, args.save, alpha, zeta)
    
    # Saving model
    if args.save is not None:
        to_save_model = model.module
        # to_save_model = model.module
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(to_save_model, path)
        
    return model.module


# +
def print_metrics(metrics, nb_batches, epoch):
    pretty_metrics = ", ".join(
        "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
        for metric_name, metric_value in metrics.items()
    )

    logger.info(f"Epoch {epoch}: {pretty_metrics}")
    
def freeze_prompt(model, dataset, text_tokens):
    model.visual_prompt.record_count = True
    model.text_prompt.record_count = True
    data_loader = dataset.train_loader
                
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits_per_image = model(inputs, text_tokens, -1, is_train=False)[0]

    model.visual_prompt.reset_counter()
    model.text_prompt.reset_counter()
    model.visual_prompt.record_count = False
    model.text_prompt.record_count = False


