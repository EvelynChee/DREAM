from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn

def scale_cam_image(cam, target_size=None):
    result = []
    cam = cam - np.expand_dims(np.min(cam.reshape(cam.shape[0],-1), axis=1), axis=list(range(1,len(cam.shape))))
    cam = cam / (1e-7 + np.expand_dims(np.max(cam.reshape(cam.shape[0],-1), axis=1), axis=list(range(1,len(cam.shape)))))

    if target_size is not None:
        for img in cam:
    #         img = img - np.min(img)
    #         img = img / (1e-7 + np.max(img))        
            if len(img.shape) > 2:
                img = zoom(np.float32(img), [
                           (t_s / i_s) for i_s, t_s in zip(img.shape, target_size[::-1])])
            else:
                img = cv2.resize(np.float32(img), target_size)

            result.append(img)
        result = np.float32(result)
    else:
        result = np.float32(cam)
        
    return result


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform, detach=True):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.detach = detach
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.detach:
            if self.reshape_transform is not None:
                activation = self.reshape_transform(activation)
            self.activations.append(activation.cpu().detach())
        else:
            self.activations.append(activation)

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.detach:
                if self.reshape_transform is not None:
                    grad = self.reshape_transform(grad)
                self.gradients = [grad.cpu().detach()] + self.gradients
            else:
                self.gradients = [grad] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# +
class GradCAM:
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        
        self.model = model.eval()
        self.target_layers = target_layers

        # Use the same device as the model.
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = False
        self.uses_gradients = True
#         self.tta_transforms = tta.Compose(
#             [
#                 tta.HorizontalFlip(),
#                 tta.Multiply(factors=[0.9, 1, 1.1]),
#             ]
#         )
        
        self.detach = True
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform, self.detach)


    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))
        
        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))
        
        else:
            raise ValueError("Invalid grads shape." 
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False, return_target_size: bool = True
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

#         if self.compute_input_gradient:
#             input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            targets = np.argmax(outputs.cpu().data.numpy(), axis=-1)
#             targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([output[target] for target, output in zip(targets, outputs)])
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                # keep the computational graph, create_graph = True is needed for hvp
                torch.autograd.grad(loss, input_tensor, retain_graph = True, create_graph = True)
                # When using the following loss.backward() method, a warning is raised: "UserWarning: Using backward() with create_graph=True will create a reference cycle"
                # loss.backward(retain_graph=True, create_graph=True)
#             if 'hpu' in str(self.device):
#                 self.__htcore.mark_step()

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth, return_target_size)
        return self.aggregate_multi_layers(cam_per_layer)
    
    
    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool, return_target_size: bool
    ) -> np.ndarray:
        if self.detach:
            activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a for a in self.activations_and_grads.activations]
            grads_list = [g for g in self.activations_and_grads.gradients]
        if return_target_size:
            target_size = self.get_target_width_height(input_tensor)
        else: 
            target_size = None

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if self.model.model.prompt_pool is not None:
#             if self.model.model.visual_prompt is not None:
                if 3*i+2 < len(activations_list):
                    layer_activations = activations_list[3*i+2]
                elif 2*i+1 < len(activations_list):
                    layer_activations = activations_list[2*i+1]
            else:
                if i < len(activations_list):
                    layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]
                
            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().detach().numpy()
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

#         if eigen_smooth:
#             cam = get_2d_projection(weighted_activations)
#         else:
        cam = weighted_activations.sum(axis=1)
        return cam
    
    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)
    
    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        if len(input_tensor.shape) == 4:
            width, height = input_tensor.size(-1), input_tensor.size(-2)
            return width, height
        elif len(input_tensor.shape) == 5:
            depth, width, height = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
            return depth, width, height
        else:
            raise ValueError("Invalid input_tensor shape. Only 2D or 3D images are supported.")
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False, 
        return_target_size: bool = True
    ) -> np.ndarray:
#         # Smooth the CAM result with test time augmentation
#         if aug_smooth is True:
#             return self.forward_augmentation_smoothing(input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor, targets, eigen_smooth, return_target_size)
    
    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
