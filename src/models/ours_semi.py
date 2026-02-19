import gco
import collections
import logging
import copy
import os
import numpy as np
import clip.clip as clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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
        
#         get_prompt(model, args, val_preprocess, i)
        
        base_model = copy.deepcopy(model)
        base_model.eval()
        
#         if i == 4:
#             evaluate(model, args, val_preprocess)
#             break

# +
from .. import datasets
from ..datasets.common import get_dataloader, maybe_dictionarize

def get_prompt(image_classifier, args, val_preprocess, task_id, test_novelty=True):
    if args.eval_datasets is None:
        return
    
    dataset_name = args.eval_datasets[task_id]
    
    logger.info(f"Evaluating on {dataset_name}")
    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(
        val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None
    )

    q = []
    for i, data in enumerate(dataloader):
        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        q.append(image_classifier.get_query(images).detach().cpu().numpy())

    q = np.concatenate(q)
    
    np.save(f'queries/avg_t{task_id}.npy',q)
    


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
    
    if load_path is not None and os.path.exists(f'{load_path}/novel_sim_t{task_id}.npy'):
        all_mix_sim, all_ori_sim, weights = np.load(f'{load_path}/novel_sim_t{task_id}.npy')
    else:
        model.eval()
        
#         if len(dataset.train_loader) > len(dataset.train_loader_nolabel):
        data_loader1 = dataset.train_loader
        data_loader2 = dataset.train_loader_nolabel
        more_nolabel = False
#         else:
#             data_loader1 = dataset.train_loader_nolabel
#             data_loader2 = dataset.train_loader
#             more_nolabel = True
    
        data_iter2 = iter(data_loader2)
        
        current_model = ImageClassifier(model, text_tokens).cuda()
        current_model.eval()
        target_layers = [current_model.model.visual.transformer.resblocks[-1].ln_1]

        cam = gradcam.GradCAM(model=current_model, target_layers=target_layers, reshape_transform=reshape_transform)

        weights = []

        all_ori_sim = []
        all_mix_sim = []
        for batch1 in data_loader1:
            try:
                batch2 = next(data_iter2)
            except:
                data_iter2 = iter(data_loader2)
                batch2 = next(data_iter2)
                
            inputs1 = batch1[0].cuda()
            inputs2 = batch2[0].cuda()
                                
#             if more_nolabel:
#                 gcam1 = cam(input_tensor=inputs1, targets=None)
#                 gcam2 = cam(input_tensor=inputs2, targets=batch2[1].cuda())
#             else:
            gcam1 = cam(input_tensor=inputs1, targets=batch1[1].cuda())
            gcam2 = cam(input_tensor=inputs2, targets=None)
            
            gcam = np.concatenate([gcam1, gcam2])
            inputs = torch.cat([inputs1, inputs2])
            
            mix_inputs, mask, _ = guidedmix(gcam, inputs, alpha, zeta, random=True)
#             mix_inputs, mask, _ = guidedmix(gcam, inputs, alpha, zeta, random=True)
# #             mix_inputs = mix_inputs #+ torch.randn_like(mix_inputs) * 0.1
            lam = (1-mask).view(inputs.size()[0], -1).sum(dim=1, keepdim=True) / (inputs.size()[-1] * inputs.size()[-2])

            with torch.no_grad():
                all_mix_sim.append(model.get_sim(mix_inputs).cpu().numpy())
                all_ori_sim.append(model.get_sim(inputs).cpu().numpy())            

                weights.append(lam.cpu().numpy())

                if flip:
                    all_mix_sim.append(model.get_sim(torch.flip(mix_inputs, dims=[3])).cpu().numpy())
                    all_ori_sim.append(model.get_sim(torch.flip(inputs, dims=[3])).cpu().numpy())            
                    weights.append(lam.cpu().numpy())        
                
        all_mix_sim = np.concatenate(all_mix_sim)
        all_ori_sim = np.concatenate(all_ori_sim)
        weights = np.concatenate(weights)[:,0]
                    
        del current_model
        del cam
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)        
        np.save(f'{save_path}/novel_sim_t{task_id}.npy', np.stack([all_mix_sim, all_ori_sim, weights]))
    
    weights = np.minimum(1-weights, weights) 

    threshold_sim = np.sum(weights * all_mix_sim)/np.sum(weights)
#     threshold_sim = 1-np.percentile(1-all_ori_sim, 60)
    
    model.prompt_pool.set_novelty_threshold(threshold_sim, task_id)
    logger.info(f'Novelty threshold: {model.prompt_pool.nov_threshold}')
    return

def guidedmix(gcam1, image1, alpha, zeta, gcam2=None, image2=None, random=True):
    B1, _, H, W = image1.size()
    
    if image2 is not None:
        B2, _, H, W = image2.size()

    if not isinstance(gcam1, torch.Tensor):
        gcam1 = torch.tensor(gcam1).to(image1.device).unsqueeze(1)
        if gcam2 is not None:
            gcam2 = torch.tensor(gcam2).to(image2.device).unsqueeze(1)
    
    if gcam2 is not None:
        sim = gcam1.view(B1, -1) @ gcam2.view(B2, -1).T
    else:
        sim = gcam1.view(B1, -1) @ gcam1.view(B1, -1).T
    sim = sim / (H*W)
    
    if random:
        sim = sim + torch.rand_like(sim) * 0.1
    
    if gcam2 is None:
        sim.fill_diagonal_(0)
        
    index = torch.argsort(sim, dim=-1, descending=True)[:,0]
    
    mask = torch.sigmoid(alpha * (gcam1.clamp(0,1)**zeta - 0.5))
    
    if image2 is not None:
        mix_image = (1-mask) * image1 + mask * image2[index]
    else:
        mix_image = (1-mask) * image1 + mask * image1[index]
    return mix_image, mask, index 

def othermix(gcam1, image1, alpha, zeta, gcam2=None, image2=None, random=True, label=None, label2=None, model=None, text=None):
    B1, _, H, W = image1.size()
    
    if image2 is not None:
        B2, _, H, W = image2.size()

    ###### puzzlemix
    input_var = Variable(image1, requires_grad=True)
    outputs = model(input_var, text)[0]
    
    if label is not None:
        target_var = Variable(label)
    else:
        target_var = Variable(torch.argmax(outputs.detach(), dim=1))
    n_classes = outputs.shape[1]
    loss_batch = 2  * nn.CrossEntropyLoss(reduction='none').cuda()(outputs, target_var) / n_classes       

    loss_batch_mean = torch.mean(loss_batch, dim=0)
    loss_batch_mean.backward(retain_graph=True)        

    unary = torch.sqrt(torch.mean(input_var.grad**2, dim=1))    
    if image2 is None:
        return mixup_process(image1, label, unary)   
    else:
        input2_var = Variable(image2, requires_grad=True)
        outputs2 = model(input2_var, text)[0]
        if label2 is not None:
            target2_var = Variable(label2)
        else:
            target2_var = Variable(torch.argmax(outputs2.detach(), dim=1))
        
        loss_batch2 = 2  * nn.CrossEntropyLoss(reduction='none').cuda()(outputs2, target2_var) / n_classes       

        loss_batch2_mean = torch.mean(loss_batch2, dim=0)
        loss_batch2_mean.backward(retain_graph=True)        

        unary2 = torch.sqrt(torch.mean(input2_var.grad**2, dim=1))    
        return mixup_process(image1, label, unary, image2, unary2)   
    
#     ############ cutmix
#     if image2 is None:
#         index = torch.randperm(B1).cuda()
#     else:
#         index = torch.randperm(B2).cuda()
#         if B1 > B2:
#             while len(index) < B1:
#                 index = torch.cat([index, torch.randperm(B2).cuda()[:B1-len(index)]])
#         elif B1 < B2:
#             index = index[:B1]
    
    
#     lam = np.random.beta(1.0, 1.0)
#     bbx1, bby1, bbx2, bby2 = rand_bbox(W, H, lam)
#     new_image = image1.clone()
#     if image2 is None:
#         new_image[:, :, bbx1:bbx2, bby1:bby2] = image1[index, :, bbx1:bbx2, bby1:bby2]
#     else:
#         new_image[:, :, bbx1:bbx2, bby1:bby2] = image2[index, :, bbx1:bbx2, bby1:bby2]
#     # adjust lambda to exactly match pixel ratio
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
#     lam = torch.ones(B1,1).to(image1.device) * lam

#     return new_image, lam, index

#     lam = np.random.beta(10, 10)
# #         lam = np.random.beta(20, 20)

#     if image2 is None:
#         new_image = lam * image1 + (1 - lam) * image1[index] 
#     else:
#         new_image = lam * image1 + (1 - lam) * image2[index] 
        
#     lam = torch.ones(B1,1).to(image1.device) * lam
#     return new_image, lam, index


def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2   

# def saliency_bbox(img, lam):
# # def saliency_bbox(img, rand_index, lam):
# #     size = img[0].size()
#     size = img.size()
#     W = size[1]
#     H = size[2]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.int(W * cut_rat)
#     cut_h = np.int(H * cut_rat)
    
#     # initialize OpenCV's static fine grained saliency detector and compute the saliency map
# #     temp_img = img.cpu().numpy().transpose(0, 2, 3, 1)
#     img_arr = img.cpu().numpy().transpose(1, 2, 0)
#     saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    
# #     lam = []
# #     new_image = img.clone()
# #     # adjust lambda to exactly match pixel ratio
# #     for i, idx in enumerate(rand_index):
# #         img_arr = temp_img[idx]
#     img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
#     img_arr = (img_arr * 255).astype(np.uint8)
#     (success, saliencyMap) = saliency.computeSaliency(img_arr)
#     saliencyMap = (saliencyMap * 255).astype("uint8")

#     maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
#     x = maximum_indices[0]
#     y = maximum_indices[1]

#     bbx1 = np.clip(x - cut_w // 2, 0, W)
#     bby1 = np.clip(y - cut_h // 2, 0, H)
#     bbx2 = np.clip(x + cut_w // 2, 0, W)
#     bby2 = np.clip(y + cut_h // 2, 0, H)

#     return bbx1, bby1, bbx2, bby2    

def gaussian_kernel(kernel_size, rand_w, rand_h, sigma):
    s = kernel_size * 2
    x_cord = torch.arange(s)
    x_grid = x_cord.repeat(s).view(s, s)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).cuda().float()
    xy_grid = torch.roll(xy_grid, rand_w, 0)
    xy_grid = torch.roll(xy_grid, rand_h, 1)
    crop_size = s // 4
    xy_grid = xy_grid[crop_size: s - crop_size, crop_size: s - crop_size]

    mean = (s - 1) / 2
    var = sigma ** 2
    g_filter = torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * var))
    g_filter = g_filter.view(kernel_size, kernel_size)

    return g_filter

def mixup_process(out, target_reweighted, grad, out2=None, grad2=None):
    block_num = 2**np.random.randint(1, 5)
#     indices = np.random.permutation(out.size(0))

    B1, _, H, W = out.size()
    if out2 is None:        
        indices = torch.randperm(B1).cuda()
        out2 = out[indices].clone()
        grad2 = grad[indices].clone()
    else:
        B2, _, H, W = out2.size()
        indices = torch.randperm(B2).cuda()
        if B1 > B2:
            while len(indices) < B1:
                indices = torch.cat([indices, torch.randperm(B2).cuda()[:B1-len(indices)]])
        elif B1 < B2:
            indices = indices[:B1]
        out2 = out2[indices].clone()
        grad2 = grad2[indices].clone()
            
    lam = np.random.beta(1.0, 1.0)

    # PuzzleMix
    out, ratio = mixup_graph(out,
                             grad,
                             out2,
                             grad2,
                             block_num=block_num,
                             alpha=lam,
                             beta=1.2,
                             gamma=0.5,
                             eta=0.2,
                             neigh_size=2,
                             n_labels=3,
                             mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).cuda(),
                             std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1).cuda(),
                             transport=True,
                             t_eps=0.8,
                             t_size=4,
                             device='cuda')
    return out, ratio.unsqueeze(-1), indices

def mixup_graph(input1,
                grad1,
                input2,
                grad2,
                block_num=2,
                alpha=0.5,
                beta=0.,
                gamma=0.,
                eta=0.2,
                neigh_size=2,
                n_labels=2,
                mean=None,
                std=None,
                transport=False,
                t_eps=10.0,
                t_size=16,
                noise=None,
                adv_mask1=0,
                adv_mask2=0,
                device='cuda',
                mp=None):
    '''Puzzle Mix'''
#     input2 = input2[indices].clone()

    batch_size, _, _, width = input1.shape
    block_size = width // block_num
    neigh_size = min(neigh_size, block_size)
#     t_size = min(t_size, block_size)

    # normalize
    beta = beta / block_num / 16

    # unary term
    grad1_pool = F.avg_pool2d(grad1, block_size)
    unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1)
#     unary2_torch = unary1_torch[indices]
    grad2_pool = F.avg_pool2d(grad2, block_size)
    unary2_torch = grad2_pool / grad2_pool.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1)

    # calculate pairwise terms
    input1_pool = F.avg_pool2d(input1 * std + mean, neigh_size)
#     input2_pool = input1_pool[indices]
    input2_pool = F.avg_pool2d(input2 * std + mean, neigh_size)

    pw_x = torch.zeros([batch_size, 2, 2, block_num - 1, block_num], device=device)
    pw_y = torch.zeros([batch_size, 2, 2, block_num, block_num - 1], device=device)

    k = block_size // neigh_size

    pw_x[:, 0, 0], pw_y[:, 0, 0] = neigh_penalty(input2_pool, input2_pool, k)
    pw_x[:, 0, 1], pw_y[:, 0, 1] = neigh_penalty(input2_pool, input1_pool, k)
    pw_x[:, 1, 0], pw_y[:, 1, 0] = neigh_penalty(input1_pool, input2_pool, k)
    pw_x[:, 1, 1], pw_y[:, 1, 1] = neigh_penalty(input1_pool, input1_pool, k)

    pw_x = beta * gamma * pw_x
    pw_y = beta * gamma * pw_y

    # re-define unary and pairwise terms to draw graph
    unary1 = unary1_torch.clone()
    unary2 = unary2_torch.clone()

    unary2[:, :-1, :] += (pw_x[:, 1, 0] + pw_x[:, 1, 1]) / 2.
    unary1[:, :-1, :] += (pw_x[:, 0, 1] + pw_x[:, 0, 0]) / 2.
    unary2[:, 1:, :] += (pw_x[:, 0, 1] + pw_x[:, 1, 1]) / 2.
    unary1[:, 1:, :] += (pw_x[:, 1, 0] + pw_x[:, 0, 0]) / 2.

    unary2[:, :, :-1] += (pw_y[:, 1, 0] + pw_y[:, 1, 1]) / 2.
    unary1[:, :, :-1] += (pw_y[:, 0, 1] + pw_y[:, 0, 0]) / 2.
    unary2[:, :, 1:] += (pw_y[:, 0, 1] + pw_y[:, 1, 1]) / 2.
    unary1[:, :, 1:] += (pw_y[:, 1, 0] + pw_y[:, 0, 0]) / 2.

    pw_x = (pw_x[:, 1, 0] + pw_x[:, 0, 1] - pw_x[:, 1, 1] - pw_x[:, 0, 0]) / 2
    pw_y = (pw_y[:, 1, 0] + pw_y[:, 0, 1] - pw_y[:, 1, 1] - pw_y[:, 0, 0]) / 2

    unary1 = unary1.detach().cpu().numpy()
    unary2 = unary2.detach().cpu().numpy()
    pw_x = pw_x.detach().cpu().numpy()
    pw_y = pw_y.detach().cpu().numpy()

    # solve graphcut
    if mp is None:
        mask = []
        for i in range(batch_size):
            mask.append(
                graphcut_multi(unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))
    else:
        input_mp = []
        for i in range(batch_size):
            input_mp.append((unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))
        mask = mp.starmap(graphcut_multi, input_mp)

    # optimal mask
    mask = torch.tensor(mask, dtype=torch.float32, device=device)
    mask = mask.unsqueeze(1)

#     # add adversarial noise
#     if adv_mask1 == 1.:
#         input1 = input1 * std + mean + noise
#         input1 = torch.clamp(input1, 0, 1)
#         input1 = (input1 - mean) / std

#     if adv_mask2 == 1.:
#         input2 = input2 * std + mean + noise[indices]
#         input2 = torch.clamp(input2, 0, 1)
#         input2 = (input2 - mean) / std

    # tranport
    if transport:
#         if t_size == -1:
#             t_block_num = block_num
#             t_size = block_size
#         elif t_size < block_size:
#             # block_size % t_size should be 0
#             t_block_num = width // t_size
#             mask = F.interpolate(mask, size=t_block_num)
#             grad1_pool = F.avg_pool2d(grad1, t_size)
#             unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(
#                 batch_size, 1, 1)
#             unary2_torch = unary1_torch[indices]
#         else:
#             t_block_num = block_num

        # input1
        plan1 = mask_transport(mask, unary1_torch, eps=t_eps)
        plan2 = mask_transport(1 - mask, unary2_torch, eps=t_eps)
        
        t_batch_size = 16
        for i in range((batch_size - 1) // t_batch_size + 1):
            idx_from = i * t_batch_size
            idx_to = min((i + 1) * t_batch_size, batch_size)
            input1[idx_from:idx_to] = transport_image(input1[idx_from:idx_to],
                                                      plan1[idx_from:idx_to], block_num,
                                                      block_size)
            input2[idx_from:idx_to] = transport_image(input2[idx_from:idx_to],
                                                      plan2[idx_from:idx_to], block_num,
                                                      block_size)
                    
#         input1 = transport_image(input1, plan, block_num, block_size)
#         input1 = transport_image(input1, plan, batch_size, t_block_num, t_size)

        # input2
#         input2 = transport_image(input2, plan, block_num, block_size)
#         input2 = transport_image(input2, plan, batch_size, t_block_num, t_size)

    # final mask and mixed ratio
    mask = F.interpolate(mask, size=width)
    ratio = mask.reshape(batch_size, -1).mean(-1)

    return mask * input1 + (1 - mask) * input2, ratio

def graphcut_multi(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2, eps=1e-8):
    '''alpha-beta swap algorithm'''
    block_num = unary1.shape[0]
    large_val = 1000 * block_num**2

    if n_labels == 2:
        prior = np.array([-np.log(alpha + eps), -np.log(1 - alpha + eps)])
    elif n_labels == 3:
        prior = np.array([
            -np.log(alpha**2 + eps), -np.log(2 * alpha * (1 - alpha) + eps),
            -np.log((1 - alpha)**2 + eps)
        ])
    elif n_labels == 4:
        prior = np.array([
            -np.log(alpha**3 + eps), -np.log(3 * alpha**2 * (1 - alpha) + eps),
            -np.log(3 * alpha * (1 - alpha)**2 + eps), -np.log((1 - alpha)**3 + eps)
        ])

    prior = eta * prior / block_num**2
    unary_cost = (large_val * np.stack([(1 - lam) * unary1 + lam * unary2 + prior[i]
                                        for i, lam in enumerate(np.linspace(0, 1, n_labels))],
                                       axis=-1)).astype(np.int32)
    pairwise_cost = np.zeros(shape=[n_labels, n_labels], dtype=np.float32)

    for i in range(n_labels):
        for j in range(n_labels):
            pairwise_cost[i, j] = (i - j)**2 / (n_labels - 1)**2

    pw_x = (large_val * (pw_x + beta)).astype(np.int32)
    pw_y = (large_val * (pw_y + beta)).astype(np.int32)

    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y,
                                      algorithm='swap') / (n_labels - 1)
    mask = labels.reshape(block_num, block_num)

    return mask



def neigh_penalty(input1, input2, k):
    '''data local smoothness term'''
    pw_x = input1[:, :, :-1, :] - input2[:, :, 1:, :]
    pw_y = input1[:, :, :, :-1] - input2[:, :, :, 1:]

    pw_x = pw_x[:, :, k - 1::k, :]
    pw_y = pw_y[:, :, :, k - 1::k]

    pw_x = F.avg_pool2d(pw_x.abs().mean(1), kernel_size=(1, k))
    pw_y = F.avg_pool2d(pw_y.abs().mean(1), kernel_size=(k, 1))

    return pw_x, pw_y

def cost_matrix(width, device='cuda'):
    '''transport cost'''
    C = np.zeros([width**2, width**2], dtype=np.float32)

    for m_i in range(width**2):
        i1 = m_i // width
        j1 = m_i % width
        for m_j in range(width**2):
            i2 = m_j // width
            j2 = m_j % width
            C[m_i, m_j] = abs(i1 - i2)**2 + abs(j1 - j2)**2

    C = C / (width - 1)**2
    C = torch.tensor(C)
    if device == 'cuda':
        C = C.cuda()

    return C

cost_matrix_dict = {
    '2': cost_matrix(2, 'cuda').unsqueeze(0),
    '4': cost_matrix(4, 'cuda').unsqueeze(0),
    '8': cost_matrix(8, 'cuda').unsqueeze(0),
    '16': cost_matrix(16, 'cuda').unsqueeze(0)
}

def mask_transport(mask, grad_pool, eps=0.01):
    '''optimal transport plan'''
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]

    n_iter = int(block_num)
    C = cost_matrix_dict[str(block_num)]

    z = (mask > 0).float()
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)

    # row and col
    for _ in range(n_iter):
        row_best = cost.min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # column resolve
        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
        plan_lose = (1 - plan_win) * plan

        cost += plan_lose

    return plan_win

# def transport_image(img, plan, batch_size, block_num, block_size):
def transport_image(img, plan, block_num, block_size):    
    '''apply transport plan to images'''
    batch_size = img.shape[0]
    input_patch = img.reshape([batch_size, 3, block_num, block_size,
                               block_num * block_size]).transpose(-2, -1)
    input_patch = input_patch.reshape([batch_size, 3, block_num, block_num, block_size,
                                       block_size]).transpose(-2, -1)
    input_patch = input_patch.reshape([batch_size, 3, block_num**2, block_size,
                                       block_size]).permute(0, 1, 3, 4, 2).unsqueeze(-1)

    input_transport = plan.transpose(
        -2, -1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(
            0, 1, 4, 2, 3)
    input_transport = input_transport.reshape(
        [batch_size, 3, block_num, block_num, block_size, block_size])
    input_transport = input_transport.transpose(-2, -1).reshape(
        [batch_size, 3, block_num, block_num * block_size, block_size])
    input_transport = input_transport.transpose(-2, -1).reshape(
        [batch_size, 3, block_num * block_size, block_num * block_size])

    return input_transport


# +
def generate_gradcam(dataset, val_preprocess, model, text_tokens, args):
    logger.info('Generating GradCAM')
    
    current_model = ImageClassifier(model, text_tokens, check_novelty=True).cuda()
    current_model.eval()
    for p in current_model.parameters():
        p.requires_grad = True

    target_layers = [current_model.model.visual.transformer.resblocks[-1].ln_1]
    
    cam = gradcam.GradCAM(model=current_model, target_layers=target_layers, reshape_transform=reshape_transform)

    all_gcams = []
    for i in range(2):
        if i == 0:
            temp_dataset = dataset.train_dataset
        else:
            temp_dataset = dataset.train_dataset_nolabel
            
#         if isinstance(temp_dataset, torch.utils.data.dataset.Subset):       
#             if isinstance(temp_dataset.dataset, torch.utils.data.dataset.Subset):       
#                 original_preprocess = temp_dataset.dataset.dataset.transform
#                 temp_dataset.dataset.dataset.transform = val_preprocess
#             else:
#                 original_preprocess = temp_dataset.dataset.transform
#                 temp_dataset.dataset.transform = val_preprocess
#         else:
#             original_preprocess = temp_dataset.transform
#             temp_dataset.transform = val_preprocess

#         data_loader = torch.utils.data.DataLoader(
#             temp_dataset,
#             batch_size=args.batch_size,
#             shuffle=False,
#             drop_last=False
#         )    
        
#         gcams = []
        
#         if i == 0:
#             for inputs, targets in data_loader:
#                 inputs, targets = inputs.cuda(), targets.cuda()
#                 gcam = cam(input_tensor=inputs, targets=targets, return_target_size=False)
#                 gcams.append(gcam)
#         else:
#             for inputs, _ in data_loader:
#                 inputs = inputs.cuda()
#                 gcam = cam(input_tensor=inputs, targets=None, return_target_size=False)
#                 gcams.append(gcam)

#         all_gcams.append(np.concatenate(gcams))
        all_gcams.append(np.zeros((len(temp_dataset),14,14)))
    
#         if i == 0:
#             if isinstance(temp_dataset, torch.utils.data.dataset.Subset):       
#                 if isinstance(temp_dataset.dataset, torch.utils.data.dataset.Subset):       
#                     dataset.train_dataset.dataset.dataset.transform = original_preprocess
#                 else:
#                     dataset.train_dataset.dataset.transform = original_preprocess
#             else:
#                 dataset.train_dataset.transform = original_preprocess
#         else:
#             if isinstance(temp_dataset, torch.utils.data.dataset.Subset):       
#                 if isinstance(temp_dataset.dataset, torch.utils.data.dataset.Subset):       
#                     dataset.train_dataset_nolabel.dataset.dataset.transform = original_preprocess
#                 else:
#                     dataset.train_dataset_nolabel.dataset.transform = original_preprocess
#             else:
#                 dataset.train_dataset_nolabel.transform = original_preprocess
            

    del current_model
    del cam
                        
    dataset.set_gradcam(all_gcams[0], all_gcams[1])
        
    return 


# -

def dataset_selection(model, dataloader, texts, task_id, k=5):
    few_shot_data = {} # create few_shot data

    labels = []    
    preds = []
    
    count = 0
    with torch.no_grad():
        for images, targets in dataloader:        
            labels.append(targets.numpy())
    
    labels = np.concatenate(labels)
    
    for cls in np.unique(labels):
        cls_idx = np.where(labels==cls)[0]
        few_shot_data[cls] = np.random.choice(cls_idx, size=k, replace=False)
        
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
    
    mix_texts = [template(f'partially abstract {x}') for x in dataset.classnames]
    mix_texts = clip.tokenize(mix_texts).cuda()

    alpha, zeta = 5, 0.5
    if args.train_dataset in ['Caltech101']:
        alpha, zeta = 10.0, 1.0

#     if task_id > 4:
#         args.load = None

        
    logger.info('=====few-shot======')
    few_shot_data_idx = dataset_selection(base_model, dataset.train_loader_noshuffle, texts, task_id, k=args.few_shot)            

    dataset.update_train_loader(few_shot_data_idx)
    
    if args.load is not None and os.path.exists(os.path.join(args.load, f"{args.train_dataset}.pth")):
        utils.torch_load(model, os.path.join(args.load, f"{args.train_dataset}.pth"))
        
#         set_novelty_threshold(model, val_preprocess, task_id, texts, dataset, args.save, alpha, zeta, args.load)
    
        return model

    data_iter = iter(dataset.train_loader)
    num_batches = len(dataset.train_loader)

    data_iter_nolabel = iter(dataset.train_loader_nolabel)
    
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
            images, labels, gradcam = next(data_iter)
            
        try:
            images_nolabel, _, gradcam_nolabel = next(data_iter_nolabel)
        except:
            data_iter_nolabel = iter(dataset.train_loader_nolabel)
            images_nolabel, _, gradcam_nolabel = next(data_iter_nolabel)
            
        images, labels, gradcam = images.cuda(), labels.cuda(), gradcam.cuda()
        images_nolabel, gradcam_nolabel = images_nolabel.cuda(), gradcam_nolabel.cuda()
        bs = images.shape[0]

        
        images2 = torch.cat([images, images_nolabel])
        logits_per_image, _, prompt_loss, image_feature, text_feature = model(images2, texts, task_id, is_train=True)  # 分开        

        pred_nolabel = logits_per_image[bs:].max(dim=1).indices.detach()
        mix_images1, lam1, mix_index1 = othermix(gradcam, images, alpha, zeta, gradcam_nolabel, images_nolabel, label=labels, label2=pred_nolabel, model=model, text=texts)
#         mix_images1, mask, mix_index1 = guidedmix(gradcam, images, alpha, zeta, gradcam_nolabel, images_nolabel)
#         lam1 = (1-mask).view(images.size()[0], -1).sum(dim=1, keepdim=True) / (images.size()[-1] * images.size()[-2])
#         lam1 = lam1.squeeze(dim=1)
        mix_images2, lam2, mix_index2 = othermix(gradcam_nolabel, images_nolabel, alpha, zeta, gradcam, images, label=pred_nolabel, label2=labels, model=model, text=texts)
#         mix_images2, mask, mix_index2 = guidedmix(gradcam_nolabel, images_nolabel, alpha, zeta, gradcam, images)
#         lam2 = mask.view(images_nolabel.size()[0], -1).sum(dim=1, keepdim=True) / (images.size()[-1] * images.size()[-2])
#         lam2 = lam2.squeeze(dim=1)
        
        mix_images = torch.cat([mix_images1, mix_images2])
        lam = torch.cat([lam1, 1-lam2])
        
        if (iteration + 1) > (total_iterations // 2):
            pred_nolabel = logits_per_image[bs:].max(dim=1).indices.detach()
            
            mix_labels1 = len(dataset.classnames) * labels + pred_nolabel[mix_index1]
            mix_labels2 = len(dataset.classnames) * labels[mix_index2] + pred_nolabel
            mix_labels = torch.cat([mix_labels1, mix_labels2])
            mix_texts_sub = mix_texts[mix_labels.unique()]
            mix_labels = (mix_labels.unsqueeze(dim=1) == mix_labels.unique().unsqueeze(dim=0)).nonzero()[:,1]
        
            mix_logits_per_image, _, mix_prompt_loss, _, _ = model(mix_images, mix_texts_sub, task_id, is_train=True)  # 分开        
        else:
            mix_labels = torch.cat([labels, labels[mix_index2]])
            
            mix_logits_per_image, _, mix_prompt_loss, _, _ = model(mix_images, mix_texts, task_id, is_train=True)  # 分开                    
            
        # -- cross entropy loss --
        loss = F.cross_entropy(logits_per_image[:bs], labels, label_smoothing=args.ls)
        metrics["clf_loss"] += loss.item()          

        prompt_loss = prompt_loss.mean() + (prompt_loss.mean() - mix_prompt_loss.mean()) 
        loss += prompt_loss
        metrics["aux_loss"] += prompt_loss.item()   
        
        mix_loss = ((lam**2) * F.cross_entropy(mix_logits_per_image, mix_labels, reduction='none', label_smoothing=args.ls)).mean() #* 0.1 
        loss += mix_loss        
        metrics["mixclf_loss"] += mix_loss.item()       
                
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics["acc"] += (labels == logits_per_image[:bs].argmax(axis=1)).float().mean().item()

        # evaluation
        if (iteration + 1) % num_batches == 0:
            print_metrics(metrics, num_batches, (iteration + 1) // num_batches)
            metrics = collections.defaultdict(float)

        if (iteration + 1) == (total_iterations // 2):
            mix_texts = [template(f'partially abstract {x1}, mixed with a {x2}' if x1 != x2 else f'partially abstract {x1}, mixed with another {x1}') for x1 in dataset.classnames for x2 in dataset.classnames]
            mix_texts = clip.tokenize(mix_texts).cuda()

            
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


