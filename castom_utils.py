import yaml
import time
from pathlib import Path
from typing import List, Optional

from matplotlib import font_manager
from fontTools.ttLib import TTFont

import torch
from torchvision.transforms import v2


class Ansi:
    green = '\033[32m'
    red = '\033[31m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'

#------------------
# torch
#------------------
class DeNormalize(v2.Normalize):
    """
    Функция для денормализации изображений.
    """
    def __init__(self,mean, std, *args, **kwargs):
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

def save_model(model, optimizer, scheduler, path, filename):
    """
    Сохранение весов моделей torch.
    """
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
# Config
#------------------
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_config(path: str = 'config.yaml'):
    """
    Загрузка конфига.
    """
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.loader.BaseLoader)

    # Значения, ключи которых начинаются на 'path' оборачиваем в функцию Path из pathlib
    for key, value in config.items():
        if key.startswith('path_'):
            config[key] = Path(value)

    # Формируем строку со всеми включенными символами
    vocab = (sorted(config['vocab_category'].items()))
    config['vocab'] = ''.join(v for _, v in vocab)
    config['vocab_unicode'] = list(map(ord, config['vocab']))

    # Делаем возможным обращение к элементам словаря через атрибуты
    config = AttrDict(config)
    
    print(f'{Ansi.green}{Ansi.bold}Config is loaded!{Ansi.end}')
    print(f'A list of all characters used:')
    print(config.vocab)

    return config

    
#------------------
# fonts
#------------------
def _is_supported_chars(font_path, vocab_unicode: List[int]):
    """
    Проверяет, поддерживает ли шрифт все нужные символы
    """
    try:
        font = TTFont(font_path)
        cmap = font['cmap'].getBestCmap()
        for code in vocab_unicode:
            if code not in cmap:
                return False
        return True
    except:
        return False


def valid_fonts(fonts_list: List[str], vocab_unicode: List[int]):
    """
    Отбирает из списка шрифты, поддеривающие все необходимые символы
    """
    return [font for font in fonts_list if _is_supported_chars(font, vocab_unicode)]


def get_system_fonts():
    """
    Возвращает список системных шрифтов
    """
    # Получаем список всех шрифтов, установленных в системе
    fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')  # Системные шрифты
    return fonts


def get_fonts_from_folder(path: str):
    """
    Возвращает список шрифтов из папки
    """
    path = Path(path)
    fonts = path.rglob('*.ttf')
    return [str(font) for font in fonts]