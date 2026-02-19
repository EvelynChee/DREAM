import os
import re

import numpy as np
import torch
from torchvision import datasets

from .cifar10 import CIFAR10 as cifar10, CIFAR100 as cifar100
import PIL
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, RandomHorizontalFlip


def underline_to_space(s):
    return s.replace("_", " ")


# +
class ClassificationDataset:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ) -> None:
        self.name = "classification_dataset"
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        if batch_size_eval is None:
            self.batch_size_eval = batch_size
        else:
            self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.append_dataset_name_to_template = append_dataset_name_to_template

        self.train_dataset = self.test_dataset = None
        self.train_loader = self.test_loader = None
        self.classnames = None
        self.templates = None

        self.label_index = []
        
    def build_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
        self.train_loader_noshuffle = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
    def set_gradcam(self, gcam, gcam_nolabel=None):
        if isinstance(self.train_dataset, torch.utils.data.dataset.Subset):            
            if isinstance(self.train_dataset.dataset, torch.utils.data.dataset.Subset):            
                subset_gcam = np.zeros((len(self.train_dataset.dataset.dataset), gcam.shape[1], gcam.shape[2]))
                subset_gcam[[self.train_dataset.dataset.indices[idx] for idx in self.train_dataset.indices]] = gcam
                self.train_dataset.dataset.dataset.gradcam = subset_gcam
            else:
                subset_gcam = np.zeros((len(self.train_dataset.dataset), gcam.shape[1], gcam.shape[2]))
                subset_gcam[self.train_dataset.indices] = gcam
                self.train_dataset.dataset.gradcam = subset_gcam
        else:
            self.train_dataset.gradcam = gcam
            
        if gcam_nolabel is not None:
            if isinstance(self.train_dataset, torch.utils.data.dataset.Subset):            
                if isinstance(self.train_dataset.dataset, torch.utils.data.dataset.Subset):            
                    subset_gcam = np.zeros((len(self.train_dataset_nolabel.dataset.dataset), gcam.shape[1], gcam.shape[2]))
                    subset_gcam[[self.train_dataset_nolabel.dataset.indices[idx] for idx in self.train_dataset_nolabel.indices]] = gcam_nolabel
                    self.train_dataset_nolabel.dataset.dataset.gradcam = subset_gcam
                else:
                    subset_gcam = np.zeros((len(self.train_dataset_nolabel.dataset), gcam.shape[1], gcam.shape[2]))
                    subset_gcam[self.train_dataset_nolabel.indices] = gcam_nolabel
                    self.train_dataset_nolabel.dataset.gradcam = subset_gcam
            else:
                self.train_dataset_nolabel.gradcam = gcam_nolabel
            
    def stats(self):
        L_train = len(self.train_dataset)
        L_test = len(self.test_dataset)
        N_class = len(self.classnames)
        return L_train, L_test, N_class

    @property
    def template(self):
        if self.append_dataset_name_to_template:
            return lambda x: self.templates[0](x)[:-1] + f", from dataset {self.name}]."
        return self.templates[0]

    def process_labels(self):
        self.classnames = [underline_to_space(x) for x in self.classnames]

    def split_dataset(self, dataset, ratio=0.8):
        train_size = int(ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        return train_dataset, test_dataset
    
    def update_train_loader(self, idx):
        nolabel_index = list(set(list(range(len(self.train_dataset)))) - set(idx))
        
        self.train_dataset_nolabel = torch.utils.data.Subset(self.train_dataset, nolabel_index)
        self.train_loader_nolabel = torch.utils.data.DataLoader(
            self.train_dataset_nolabel,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, idx)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

#     def label_data(self, label):
#         self.label_index = list(set(self.label_index + label))
        
#         self.train_loader = torch.utils.data.DataLoader(
#             torch.utils.data.Subset(self.train_dataset, self.label_index),
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#         )
        
#         nolabel_index = list(set(list(range(len(self.train_dataset)))) - set(self.label_index))
#         self.train_loader_nolabel = torch.utils.data.DataLoader(
#             torch.utils.data.Subset(self.train_dataset, nolabel_index),
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#         )
        
#         print(f"Labelled data: {len(self.label_index)}/{len(self.train_dataset)}")
            
    @property
    def class_to_idx(self):
        return {v: k for k, v in enumerate(self.classnames)}


# +
class PyTorchFGVCAircraft(datasets.FGVCAircraft):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradcam = None
        
    def __getitem__(self, idx):
        if self.gradcam is not None:
            image_file, label = self._image_files[idx], self._labels[idx]
            gradcam = self.gradcam[idx]
            image = PIL.Image.open(image_file).convert("RGB")       
            
            gradcam = cv2.resize(gradcam, image.size)
            
            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)
        
            return image, label, gradcam
        else:
            image, label = super().__getitem__(idx)
            return image, label
            
class Aircraft(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "aircraft"
        self.train_dataset = PyTorchFGVCAircraft(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = PyTorchFGVCAircraft(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of aircraft.",
            lambda c: f"a photo of the {c}, a type of aircraft.",
        ]

    # def process_labels(self):
    #     label = self.classnames
    #     for i in range(len(label)):
    #         if label[i].startswith("7"):
    #             label[i] = "Boeing " + label[i]
    #         elif label[i].startswith("An") or label[i].startswith("ATR"):
    #             pass
    #         elif label[i].startswith("A"):
    #             label[i] = "Airbus " + label[i]



# +
class PyTorchCaltech101(datasets.Caltech101):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradcam = None
        
    def __getitem__(self, idx):
        if self.gradcam is not None:
            image = PIL.Image.open(
                os.path.join(
                    self.root,
                    "101_ObjectCategories",
                    self.categories[self.y[idx]],
                    f"image_{self.index[idx]:04d}.jpg",
                )
            )

            label = self.y[idx]
            gradcam = self.gradcam[idx]
            gradcam = cv2.resize(gradcam, image.size)

            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)

            return image, label, gradcam
        else:
            image, label = super().__getitem__(idx)
            return image, label
        
    
    
class Caltech101(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "caltech101"
        dataset = PyTorchCaltech101(
            self.location, download=True, transform=self.preprocess
        )
        self.classnames = dataset.categories

        train_dataset, test_dataset = self.split_dataset(dataset)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.build_dataloader()

        self.classnames = [
            "off-center face",
            "centered face",
            "leopard",
            "motorbike",
            "accordion",
            "airplane",
            "anchor",
            "ant",
            "barrel",
            "bass",
            "beaver",
            "binocular",
            "bonsai",
            "brain",
            "brontosaurus",
            "buddha",
            "butterfly",
            "camera",
            "cannon",
            "side of a car",
            "ceiling fan",
            "cellphone",
            "chair",
            "chandelier",
            "body of a cougar cat",
            "face of a cougar cat",
            "crab",
            "crayfish",
            "crocodile",
            "head of a  crocodile",
            "cup",
            "dalmatian",
            "dollar bill",
            "dolphin",
            "dragonfly",
            "electric guitar",
            "elephant",
            "emu",
            "euphonium",
            "ewer",
            "ferry",
            "flamingo",
            "head of a flamingo",
            "garfield",
            "gerenuk",
            "gramophone",
            "grand piano",
            "hawksbill",
            "headphone",
            "hedgehog",
            "helicopter",
            "ibis",
            "inline skate",
            "joshua tree",
            "kangaroo",
            "ketch",
            "lamp",
            "laptop",
            "llama",
            "lobster",
            "lotus",
            "mandolin",
            "mayfly",
            "menorah",
            "metronome",
            "minaret",
            "nautilus",
            "octopus",
            "okapi",
            "pagoda",
            "panda",
            "pigeon",
            "pizza",
            "platypus",
            "pyramid",
            "revolver",
            "rhino",
            "rooster",
            "saxophone",
            "schooner",
            "scissors",
            "scorpion",
            "sea horse",
            "snoopy (cartoon beagle)",
            "soccer ball",
            "stapler",
            "starfish",
            "stegosaurus",
            "stop sign",
            "strawberry",
            "sunflower",
            "tick",
            "trilobite",
            "umbrella",
            "watch",
            "water lilly",
            "wheelchair",
            "wild cat",
            "windsor chair",
            "wrench",
            "yin and yang symbol",
        ]

        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a painting of a {c}.",
            lambda c: f"a plastic {c}.",
            lambda c: f"a sculpture of a {c}.",
            lambda c: f"a sketch of a {c}.",
            lambda c: f"a tattoo of a {c}.",
            lambda c: f"a toy {c}.",
            lambda c: f"a rendition of a {c}.",
            lambda c: f"a embroidered {c}.",
            lambda c: f"a cartoon {c}.",
            lambda c: f"a {c} in a video game.",
            lambda c: f"a plushie {c}.",
            lambda c: f"a origami {c}.",
            lambda c: f"art of a {c}.",
            lambda c: f"graffiti of a {c}.",
            lambda c: f"a drawing of a {c}.",
            lambda c: f"a doodle of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a painting of the {c}.",
            lambda c: f"the plastic {c}.",
            lambda c: f"a sculpture of the {c}.",
            lambda c: f"a sketch of the {c}.",
            lambda c: f"a tattoo of the {c}.",
            lambda c: f"the toy {c}.",
            lambda c: f"a rendition of the {c}.",
            lambda c: f"the embroidered {c}.",
            lambda c: f"the cartoon {c}.",
            lambda c: f"the {c} in a video game.",
            lambda c: f"the plushie {c}.",
            lambda c: f"the origami {c}.",
            lambda c: f"art of the {c}.",
            lambda c: f"graffiti of the {c}.",
            lambda c: f"a drawing of the {c}.",
            lambda c: f"a doodle of the {c}.",
        ]


# +
class PyTorchMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradcam = None

        transforms = []
        for t in self.transform.transforms:
            if not isinstance(t, RandomHorizontalFlip):
                transforms.append(t)
        self.transform = Compose(transforms)
                
    def __getitem__(self, idx):
        if self.gradcam is not None:
            image, label = self.data[idx], int(self.targets[idx])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            image = PIL.Image.fromarray(image.numpy(), mode="L")
            
            gradcam = self.gradcam[idx]
            gradcam = cv2.resize(gradcam, image.size)
            
            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)

            return image, label, gradcam

        else:
            image, label = super().__getitem__(idx)
            return image, label
        
class MNIST(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "mnist"
        self.train_dataset = PyTorchMNIST(
            self.location, train=True, download=True, transform=self.preprocess
        )
        self.test_dataset = PyTorchMNIST(
            self.location, train=False, download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f'a photo of the number: "{c}".',
        ]


# -

class CIFAR10(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "cifar10"
        dataset = cifar10(preprocess=self.preprocess, location=self.location)

        self.train_dataset = dataset.train_dataset
        self.test_dataset = dataset.test_dataset
        self.build_dataloader()
        self.classnames = dataset.classnames
        self.process_labels()
        self.templates = dataset.template


# +
class PyTorchCIFAR100(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradcam = None
        
    def __getitem__(self, idx):
        if self.gradcam is not None:
            image, label = self.data[idx], self.targets[idx]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            image = PIL.Image.fromarray(image)
            
            gradcam = self.gradcam[idx]
            gradcam = cv2.resize(gradcam, image.size)
            
            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)

            return image, label, gradcam

        else:
            image, label = super().__getitem__(idx)
            return image, label

class CIFAR100(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "cifar100"

        self.train_dataset = PyTorchCIFAR100(
            root=self.location, download=True, train=True, transform=self.preprocess
        )
        self.test_dataset = PyTorchCIFAR100(
            root=self.location, download=True, train=False, transform=self.preprocess
        )

        self.classnames = self.test_dataset.classes
        self.templates = [
            lambda c : f'a photo of a {c}.',
            lambda c : f'a blurry photo of a {c}.',
            lambda c : f'a black and white photo of a {c}.',
            lambda c : f'a low contrast photo of a {c}.',
            lambda c : f'a high contrast photo of a {c}.',
            lambda c : f'a bad photo of a {c}.',
            lambda c : f'a good photo of a {c}.',
            lambda c : f'a photo of a small {c}.',
            lambda c : f'a photo of a big {c}.',
            lambda c : f'a photo of the {c}.',
            lambda c : f'a blurry photo of the {c}.',
            lambda c : f'a black and white photo of the {c}.',
            lambda c : f'a low contrast photo of the {c}.',
            lambda c : f'a high contrast photo of the {c}.',
            lambda c : f'a bad photo of the {c}.',
            lambda c : f'a good photo of the {c}.',
            lambda c : f'a photo of the small {c}.',
            lambda c : f'a photo of the big {c}.',
        ]
        
#         dataset = cifar100(preprocess=self.preprocess, location=self.location)
#         self.train_dataset = dataset.train_dataset
#         self.test_dataset = dataset.test_dataset
        self.build_dataloader()
#         self.classnames = dataset.classnames
        self.process_labels()
#         self.templates = dataset.template


# +
class PyTorchDTD(datasets.DTD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.gradcam = None
        
    def __getitem__(self, idx):
        if self.gradcam is not None:
            image_file, label = self._image_files[idx], self._labels[idx]
            image = PIL.Image.open(image_file).convert("RGB")
            
            gradcam = self.gradcam[idx]
            gradcam = cv2.resize(gradcam, image.size)
            
            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)

            return image, label, gradcam

        else:
            image, label = super().__getitem__(idx)
            return image, label
        
class DTD(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "dtd"
        self.train_dataset = PyTorchDTD(
            self.location, split="train", download=False, transform=self.preprocess
        )
        self.test_dataset = PyTorchDTD(
            self.location, split="test", download=False, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f'a photo of a {c} texture.',
            lambda c: f'a photo of a {c} pattern.',
            lambda c: f'a photo of a {c} thing.',
            lambda c: f'a photo of a {c} object.',
            lambda c: f'a photo of the {c} texture.',
            lambda c: f'a photo of the {c} pattern.',
            lambda c: f'a photo of the {c} thing.',
            lambda c: f'a photo of the {c} object.',
        ]


# +
class PyTorchEuroSAT(datasets.EuroSAT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradcam = None
        
    def __getitem__(self, idx):
        if self.gradcam is not None:
            path, label = self.samples[idx]
            image = self.loader(path)
            
            gradcam = self.gradcam[idx]
            gradcam = cv2.resize(gradcam, image.size)
            
            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)

            return image, label, gradcam

        else:
            image, label = super().__getitem__(idx)
            return image, label
        
class EuroSAT(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "eurosat"
        dataset = PyTorchEuroSAT(
            self.location, download=True, transform=self.preprocess
        )
        train_dataset, test_dataset = self.split_dataset(dataset)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.build_dataloader()

        self.classnames = [
            "annual crop land",
            "forest",
            "brushland or shrubland",
            "highway or road",
            "industrial buildings or commercial buildings",
            "pasture land",
            "permanent crop land",
            "residential buildings or homes or apartments",
            "river",
            "lake or sea",
        ]

        self.templates = [
            lambda c: f"a centered satellite photo of {c}.",
            lambda c: f"a centered satellite photo of a {c}.",
            lambda c: f"a centered satellite photo of the {c}.",
        ]

    def process_labels(self):
        super().process_labels()
        self.classnames = [re.sub(r"(\w)([A-Z])", r"\1 \2", x) for x in self.classnames]


# +
class PyTorchFlowers102(datasets.Flowers102):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradcam = None
        
    def __getitem__(self, idx):
        if self.gradcam is not None:
            image_file, label = self._image_files[idx], self._labels[idx]
            image = PIL.Image.open(image_file).convert("RGB")
            
            gradcam = self.gradcam[idx]
            gradcam = cv2.resize(gradcam, image.size)
            
            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)

            return image, label, gradcam

        else:
            image, label = super().__getitem__(idx)
            return image, label
        
class Flowers(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "flowers"
        self.train_dataset = PyTorchFlowers102(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = PyTorchFlowers102(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = [
            "pink primrose",
            "hard-leaved pocket orchid",
            "canterbury bells",
            "sweet pea",
            "english marigold",
            "tiger lily",
            "moon orchid",
            "bird of paradise",
            "monkshood",
            "globe thistle",
            "snapdragon",
            "colt's foot",
            "king protea",
            "spear thistle",
            "yellow iris",
            "globe-flower",
            "purple coneflower",
            "peruvian lily",
            "balloon flower",
            "giant white arum lily",
            "fire lily",
            "pincushion flower",
            "fritillary",
            "red ginger",
            "grape hyacinth",
            "corn poppy",
            "prince of wales feathers",
            "stemless gentian",
            "artichoke",
            "sweet william",
            "carnation",
            "garden phlox",
            "love in the mist",
            "mexican aster",
            "alpine sea holly",
            "ruby-lipped cattleya",
            "cape flower",
            "great masterwort",
            "siam tulip",
            "lenten rose",
            "barbeton daisy",
            "daffodil",
            "sword lily",
            "poinsettia",
            "bolero deep blue",
            "wallflower",
            "marigold",
            "buttercup",
            "oxeye daisy",
            "common dandelion",
            "petunia",
            "wild pansy",
            "primula",
            "sunflower",
            "pelargonium",
            "bishop of llandaff",
            "gaura",
            "geranium",
            "orange dahlia",
            "pink-yellow dahlia",
            "cautleya spicata",
            "japanese anemone",
            "black-eyed susan",
            "silverbush",
            "californian poppy",
            "osteospermum",
            "spring crocus",
            "bearded iris",
            "windflower",
            "tree poppy",
            "gazania",
            "azalea",
            "water lily",
            "rose",
            "thorn apple",
            "morning glory",
            "passion flower",
            "lotus",
            "toad lily",
            "anthurium",
            "frangipani",
            "clematis",
            "hibiscus",
            "columbine",
            "desert-rose",
            "tree mallow",
            "magnolia",
            "cyclamen",
            "watercress",
            "canna lily",
            "hippeastrum",
            "bee balm",
            "ball moss",
            "foxglove",
            "bougainvillea",
            "camellia",
            "mallow",
            "mexican petunia",
            "bromelia",
            "blanket flower",
            "trumpet creeper",
            "blackberry lily",
        ]
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of flower.",
        ]


# +
class PyTorchFood101(datasets.Food101):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradcam = None
        
    def __getitem__(self, idx):
        if self.gradcam is not None:
            image_file, label = self._image_files[idx], self._labels[idx]
            image = PIL.Image.open(image_file).convert("RGB")
            
            gradcam = self.gradcam[idx]
            gradcam = cv2.resize(gradcam, image.size)
            
            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)

            return image, label, gradcam

        else:
            image, label = super().__getitem__(idx)
            return image, label
        
class Food(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "food"
        self.train_dataset = PyTorchFood101(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = PyTorchFood101(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of food.",
        ]


# +
class PyTorchOxfordIIITPet(datasets.OxfordIIITPet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradcam = None
        
    def __getitem__(self, idx):
        if self.gradcam is not None:
            image = PIL.Image.open(self._images[idx]).convert("RGB")
            label = self._labels[idx]
            
            gradcam = self.gradcam[idx]
            gradcam = cv2.resize(gradcam, image.size)
            
            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)

            return image, label, gradcam

        else:
            image, label = super().__getitem__(idx)
            return image, label
        
class OxfordPet(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "oxford pet"
        self.train_dataset = PyTorchOxfordIIITPet(
            self.location, split="trainval", download=True, transform=self.preprocess
        )
        self.test_dataset = PyTorchOxfordIIITPet(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of pet.",
        ]


# +
class PyTorchStanfordCars(datasets.StanfordCars):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradcam = None
        
    def __getitem__(self, idx):
        if self.gradcam is not None:
            image_path, label = self._samples[idx]
            image = PIL.Image.open(image_path).convert("RGB")
            
            gradcam = self.gradcam[idx]
            gradcam = cv2.resize(gradcam, image.size)
            
            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)

            return image, label, gradcam

        else:
            image, label = super().__getitem__(idx)
            return image, label
        
        
class StanfordCars(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "stanford cars"
        self.train_dataset = PyTorchStanfordCars(
            self.location, split="train", download=True, transform=self.preprocess
        )
        self.test_dataset = PyTorchStanfordCars(
            self.location, split="test", download=True, transform=self.preprocess
        )
        self.build_dataloader()
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of car.",
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a photo of my {c}.",
            lambda c: f"i love my {c}!",
            lambda c: f"a photo of my dirty {c}.",
            lambda c: f"a photo of my clean {c}.",
            lambda c: f"a photo of my new {c}.",
            lambda c: f"a photo of my old {c}.",
        ]


# +
class PyTorchSUN397(datasets.SUN397):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradcam = None
        
    def __getitem__(self, idx):
        if self.gradcam is not None:
            image_file, label = self._image_files[idx], self._labels[idx]
            image = PIL.Image.open(image_file).convert("RGB")
            
            gradcam = self.gradcam[idx]
            gradcam = cv2.resize(gradcam, image.size)
            
            if self.transform:
                image = Compose(self.transform.transforms[:3])(image)
                gradcam = ToTensor()(gradcam)
                
                image_gradcam = torch.cat([image, gradcam], dim=0)
                image_gradcam = Compose(self.transform.transforms[3:])(image_gradcam)                
                
                image = image_gradcam[:3]
                gradcam = image_gradcam[-1:]

            if self.target_transform:
                label = self.target_transform(label)

            return image, label, gradcam

        else:
            image, label = super().__getitem__(idx)
            return image, label
        
class SUN397(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "sun397"
        dataset = PyTorchSUN397(
            self.location, download=True, transform=self.preprocess
        )
        train_dataset, test_dataset = self.split_dataset(dataset)
        self.classnames = dataset.classes

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.build_dataloader()
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]
