import yaml
import time
from pathlib import Path
from torchvision.transforms import v2


class Ansi:
    green = '\033[32m'
    red = '\033[31m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'


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

    for key, value in config.items():
        if key.startswith('path_'):
            config[key] = Path(value)

    config = AttrDict(config)
    config.vocab = ''.join(config.vocab_category.values())
    
    print(f'{Ansi.green}{Ansi.bold}Config is loaded!{Ansi.end}')
    print(f'A list of all characters used:')
    print(config.vocab)

    return config