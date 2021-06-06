from time import time

import imageio
import yaml
import numpy as np
from pathlib import Path

from VSR.utils.logger import get_logger
from VSR.utils.utils import get_timestamp


class Predictor:
    """Клас класифікатор для передбачення, використовуючи вхідну модель.

    Завантажує зображення та відео із вхідного каталогу, виконує класифікацію за допомогою
    моделі та зберігає результат у вихідному каталозі.
    Може отримати шлях до вагів або надати користувачу можливість обрати шлях.

    Args:
        input_dir: string, шлях до вхідного каталогу.
        output_dir: string, шлях до вихідного каталогу.
        verbose: bool.

    Attributes:
        img_extensions: список дозволених розширень зображення.
        img_ls: список файлів зображень у вхідному каталозі.
        video_extensions: список дозволених розширень відео.
        video_ls: список файлів відео у вхідному каталозі.

    Methods:
        get_predictions: виконати класифікацію файлів із вхідного каталогу
        із використанням моделі та вагів та зберегти результати у вихідному каталозі
    """

    def __init__(self, input_dir, output_dir='./data/output', verbose=True):

        self.input_dir = Path(input_dir)
        self.data_name = self.input_dir.name
        self.output_dir = Path(output_dir) / self.data_name
        self.logger = get_logger(__name__)
        if not verbose:
            self.logger.setLevel(40)
        self.img_extensions = ('.jpeg', '.jpg', '.png')  # допустимі розширення зображень
        self.img_ls = [f for f in self.input_dir.iterdir() if f.suffix in self.img_extensions]
        self.video_extensions = ('.wmv', '.mkv', '.mp4', '.avi', '.mpeg')
        self.video_ls = [f for f in self.input_dir.iterdir() if f.suffix in self.video_extensions]
        if ( (len(self.img_ls) < 1) and (len(self.video_ls) < 1)):
            self.logger.error('Коректних зображень не знайшлося (перевірте конфігураційний файл).')
            raise ValueError('Коректних зображень не знайшлося (перевірте конфігураційний файл).')
        # Створити каталог для результатів
        if not self.output_dir.exists():
            self.logger.info('Створюю вихідний каталог:\n{}'.format(self.output_dir))
            self.output_dir.mkdir(parents=True)

    def _load_weights(self):
        """ Invokes the model's load weights function if any weights are provided. """
        if self.weights_path is not None:
            self.logger.info('Завантажено ваги із \n > {}'.format(self.weights_path))
            # loading by name automatically excludes the vgg layers
            self.model.model.load_weights(str(self.weights_path))
        else:
            self.logger.error('Помилка: Шлях до вагів не вказаний (перевірте конфігураційний файл).')
            raise ValueError('Шлях до вагів не вказаний (перевірте конфігураційний файл).')

        session_config_path = self.weights_path.parent / 'session_config.yml'
        if session_config_path.exists():
            conf = yaml.load(session_config_path.read_text(), Loader=yaml.FullLoader)
        else:
            self.logger.warning('Не вдалося знайти ваги конфігурації навчання')
            conf = {}
        conf.update({'pre-trained-weights': self.weights_path.name})
        return conf

    def _make_basename(self):
        """ Поєднує назву генератора та параметри архітектури. """

        params = [self.model.name]
        for param in np.sort(list(self.model.params.keys())):
            params.append('{g}{p}'.format(g=param, p=self.model.params[param]))
        return '-'.join(params)

    def get_predictions(self, model, weights_path):
        """ Виконує покращення. """

        self.model = model
        self.weights_path = Path(weights_path)
        weights_conf = self._load_weights()
        out_folder = self.output_dir / self._make_basename() / get_timestamp()
        self.logger.info('Результати в:\n > {}'.format(out_folder))
        if out_folder.exists():
            self.logger.warning('Каталог існує, можливо файли було перезаписано')
        else:
            out_folder.mkdir(parents=True)
        if weights_conf:
            yaml.dump(weights_conf, (out_folder / 'weights_config.yml').open('w'))
        # Класифікувати та зберегти
        for img_path in self.img_ls:
            output_path = out_folder / img_path.name
            self.logger.info('Оброблюю файл \n > {}'.format(img_path))
            start = time()
            lr_img = imageio.imread(img_path)
            sr_img = self._forward_pass(lr_img)
            end = time()
            self.logger.info('Витрачений час: {}s'.format(end - start))
            self.logger.info('Результати в: {}'.format(output_path))
            imageio.imwrite(output_path, sr_img)
        for video_path in self.video_ls:
            output_path = out_folder / video_path.name
            self.logger.info('Оброблюю файл \n > {}'.format(video_path))
            start = time()
            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data()['fps']
            duration = reader.get_meta_data()['duration']
            writer = imageio.get_writer(output_path, fps=fps)
            for lr_img in reader:
                sr_img = self._forward_pass(lr_img)
                writer.append_data(sr_img)
            writer.close()
            end = time()
            self.logger.info('Витрачений час: {}s'.format(end - start))
            self.logger.info('Результати в: {}'.format(output_path))

    def _forward_pass(self, lr_img):
        if lr_img.shape[2] == 3:
            sr_img = self.model.predict(lr_img)
            return sr_img
        else:
            self.logger.error('{} не є зображенням із трьох каналів.'.format(file_path))
