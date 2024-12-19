import yaml
import json
import time
import numpy as np
from pathlib import Path
from collections.abc import Callable
from typing import Any
from PIL import Image

from matplotlib import font_manager
from fontTools.ttLib import TTFont

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from albumentations.core.transforms_interface import BasicTransform
from doctr import transforms as T


class Ansi:
    green = '\033[32m'
    red = '\033[31m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'

def write_json(path: str | Path, obj: Any) -> None:
    """Запись объекта в *.json."""
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(obj, file, indent=2, ensure_ascii=False)

def read_json(path: str | Path) -> Any:
    """Чтение *.json файла."""
    with open(path, 'r', encoding='utf-8') as file:
        obj = json.load(file)
    return obj


#------------------
# Config
#------------------
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(path: str = 'config.yaml'):
    """Загрузка конфига для проекта."""
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.loader.BaseLoader)

    # Значения, содержащие пути, оборачиваем в функцию Path из pathlib
    path_keys = ['dataset_synthetic', 'dataset_test', 'pdf_files', 'models_save', 'models_best', 'fonts']
    for key in path_keys:
        config[key] = Path(config[key])

    # Формируем строку со всеми включенными символами
    vocab = (sorted(config['vocab_category'].items()))
    config['vocab'] = ''.join(v for _, v in vocab)
    config['vocab_train'] = config['vocab'] + ''.join(config['vocab_replace'].keys())

    config['vocab_unicode'] = list(map(ord, config['vocab']))
    config['vocab_train_unicode'] = list(map(ord, config['vocab_train']))

    # Делаем возможным обращение к элементам словаря через атрибуты
    config = AttrDict(config)
    
    print(f'{Ansi.green}{Ansi.bold}Config is loaded!{Ansi.end}')
    print(f'List of all chars used ({Ansi.bold}{len(config.vocab)}{Ansi.end}):')
    print(config.vocab)

    return config


#------------------
# torch
#------------------
class DeNormalize(v2.Normalize):
    """Денормализации изображений."""
    def __init__(self,mean, std, *args, **kwargs):
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def save_model(
    model: torch.nn.Module, 
    optimizer, 
    scheduler, 
    path: Path, 
    filename: str
) -> None:
    """Сохранение весов моделей torch в файл."""
    checkpoint = {
        'info': {
            'datetime': time.ctime(),
        },
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    path.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path.joinpath(f'{filename}.pth'))


#------------------
# fonts / text
#------------------
def _is_supported_chars(font_path: str, vocab_unicode: list[int]) -> bool:
    """Проверяет, поддерживает ли шрифт все символы из vocab_unicode."""
    try:
        font = TTFont(font_path)
        cmap = font['cmap'].getBestCmap()
        for code in vocab_unicode:
            if code not in cmap:
                return False
        return True
    except:
        return False


def valid_fonts(fonts_list: list[str], vocab_unicode: list[int]) -> list[str]:
    """Отбирает из списка шрифты, поддерживающие все символы из vocab_unicode."""
    return [font for font in fonts_list if _is_supported_chars(font, vocab_unicode)]


def get_system_fonts() -> list[str]:
    """Возвращает список системных шрифтов."""
    # Получаем список всех шрифтов, установленных в системе
    fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')  # Системные шрифты
    return fonts


def get_fonts_from_folder(path: str) -> list[str]:
    """Возвращает список шрифтов из папки."""
    path = Path(path)
    fonts = path.rglob('*.ttf')
    return [str(font) for font in fonts]

def translate_text(text: str, translate: dict[str, str]) -> str:
    """ Замена всех символов в тексте, согласно словарю translate.

    Args:
        text: Исходный текст.
        translate: Словарь вида {"искомый символ": "заменить на"}.
    """
    translation_table = str.maketrans(translate)
    return text.translate(translation_table)


#------------------
# torch
#-----------------
class RecognitionDatasetCustom(Dataset):
    
    def __init__(self, path: Path, transforms=None, transforms_alb=None):
        self.path = path
        self.transforms = transforms
        self.transforms_alb = transforms_alb
        self.annotations = read_json(path / 'labels.json')
        self.img_names = list(self.annotations.keys())
        
        for img_name in self.img_names:
            if not path.joinpath('images', img_name).exists():
                raise FileNotFoundError(f"unable to locate {Path('dataset/train', img_name)}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx: int):
        img_name = self.img_names[idx]
        target = self.annotations[img_name]
        img = Image.open(self.path.joinpath('images', img_name)).convert('RGB')

        if self.transforms_alb is not None:
            img = np.array(img)
            img = self.transforms_alb(image=img)['image']

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


class DetectionDatasetCustom(RecognitionDatasetCustom):

    def __getitem__(self, idx: int):
        img_name = self.img_names[idx]
        boxes = self.annotations[img_name]['boxes']
        boxes = np.array(boxes)
        img = Image.open(self.path.joinpath('images', img_name)).convert('RGB')

        if self.transforms_alb is not None:
            img = np.array(img)
            transformed = self.transforms_alb(image=img, bboxes=boxes, labels=[0] * len(boxes))
            img, boxes = transformed['image'], transformed['bboxes']

        if self.transforms is not None:
            img = self.transforms(img)

        return img, {'words': boxes.astype(np.float32)}


class ColorShiftTorch(v2.Transform):
    def __init__(self, min_val: float = .5, p: float = .5):
        super().__init__()
        self.min_val = min_val
        self.p = p

    def _get_params(self, flat_inputs) -> dict[str, Any]:  # make_params
        apply_transform = (torch.rand(size=(1,)) < self.p).item()
        params = dict(apply_transform=apply_transform)
        return params

    def _transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor: # transform
        if params['apply_transform']:
            return self._color_shift(inpt, self.min_val)
        else:
            return inpt

    @staticmethod
    def _color_shift(img: torch.Tensor, min_val: float = .5) -> torch.Tensor:
        shift = min_val +  torch.rand(3, 1, 1) * (1 - min_val) # [min_val, 1]
        img = (255 - (255 - img) * shift).clip(0, 255).to(dtype=torch.uint8)
        return img


class ColorShiftAlb(BasicTransform):
    def __init__(
        self, 
        min_val: float = .5, 
        always_apply: bool | None = None, 
        p: float = .5
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.min_val = min_val

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return self._color_shift(img, self.min_val)

    def get_transform_init_args_names(self):
        return("min_val",)

    @property
    def targets(self):
        return {"image": self.apply}

    @staticmethod
    def _color_shift(img: np.ndarray, min_val: float = .5) -> np.ndarray:
        shift = min_val +  np.random.rand(1, 1, 3) * (1 - min_val) # [min_val, 1]
        img = (255 - (255 - img) * shift).clip(0, 255).astype(np.uint8)
        return img