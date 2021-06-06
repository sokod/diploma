import os

import imageio
import numpy as np

from VSR.utils.logger import get_logger


class DataHandler:
    """
    DataHandler генерує додаткові партії зображень для навчання та валідації.

    Args:
        lr_dir: каталог із зображеннями низької якості.
        hr_dir: каталог із еталонними зображеннями.
        patch_size: integer, розмір ділянок отриманих із зображень.
        scale: integer, коефіцієнт підвищення роздільної здатності.
        n_validation_samples: integer, розмір партії валідації.
    """

    def __init__(self, lr_dir, hr_dir, patch_size, scale, n_validation_samples=None):
        self.folders = {'hr': hr_dir, 'lr': lr_dir}  # image folders
        self.extensions = ('.png', '.jpeg', '.jpg')  # admissible extension
        self.img_list = {}  # list of file names
        self.n_validation_samples = n_validation_samples
        self.patch_size = patch_size
        self.scale = scale
        self.patch_size = {'lr': patch_size, 'hr': patch_size * self.scale}
        self.logger = get_logger(__name__)
        self._make_img_list()
        self._check_dataset()

    def _make_img_list(self):
        """ Створює хеш з лістингом lr, hr зображень із відповідних каталогів. """

        for res in ['hr', 'lr']:
            file_names = os.listdir(self.folders[res])
            file_names = [file for file in file_names if file.endswith(self.extensions)]
            self.img_list[res] = np.sort(file_names)

        if self.n_validation_samples:
            samples = np.random.choice(
                range(len(self.img_list['hr'])), self.n_validation_samples, replace=False
            )
            for res in ['hr', 'lr']:
                self.img_list[res] = self.img_list[res][samples]

    def _check_dataset(self):
        """ Перевірка датасету. """

        # the order of these asserts is important for testing
        assert len(self.img_list['hr']) == self.img_list['hr'].shape[0], 'UnevenDatasets'
        assert self._matching_datasets(), 'Input/LabelsMismatch'

    def _matching_datasets(self):
        """ Приблизне співставлення назв в директоріях lr та hr. """
        # LR_name.png = HR_name+x+scale.png
        # or
        # LR_name.png = HR_name.png
        LR_name_root = [x.split('.')[0].rsplit('x', 1)[0] for x in self.img_list['lr']]
        HR_name_root = [x.split('.')[0] for x in self.img_list['hr']]
        return np.all(HR_name_root == LR_name_root)

    def _not_flat(self, patch, flatness):
        """
        Визначає складність ділянки зображення. Поріг рівності визначає flatness.
        """

        if max(np.std(patch, axis=0).mean(), np.std(patch, axis=1).mean()) < flatness:
            return False
        else:
            return True

    def _crop_imgs(self, imgs, batch_size, flatness):
        """
        Отримує випадкові координати верхнього лівого кута в LR зображені, помножене на
        коефіцієнт збільшення для отримання координат в HR.
        Отримує batch_size + n можливих координат.
        Приймає набір лише із стандартним відхиленням інтенсивності пікселів більшим за границю, або
        якщо ділянки більше неможливо видкинути (n вже було відкинуто)
        Ріже зображення на квадратні ділянки розміром patch_size із обраного верхнього лівого кута
        """

        slices = {}
        crops = {}
        crops['lr'] = []
        crops['hr'] = []
        accepted_slices = {}
        accepted_slices['lr'] = []
        top_left = {'x': {}, 'y': {}}
        n = 50 * batch_size
        for i, axis in enumerate(['x', 'y']):
            top_left[axis]['lr'] = np.random.randint(
                0, imgs['lr'].shape[i] - self.patch_size['lr'] + 1, batch_size + n
            )
            top_left[axis]['hr'] = top_left[axis]['lr'] * self.scale
        for res in ['lr', 'hr']:
            slices[res] = np.array(
                [
                    {'x': (x, x + self.patch_size[res]), 'y': (y, y + self.patch_size[res])}
                    for x, y in zip(top_left['x'][res], top_left['y'][res])
                ]
            )

        for slice_index, s in enumerate(slices['lr']):
            candidate_crop = imgs['lr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            if self._not_flat(candidate_crop, flatness) or n == 0:
                crops['lr'].append(candidate_crop)
                accepted_slices['lr'].append(slice_index)
            else:
                n -= 1
            if len(crops['lr']) == batch_size:
                break

        accepted_slices['hr'] = slices['hr'][accepted_slices['lr']]

        for s in accepted_slices['hr']:
            candidate_crop = imgs['hr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            crops['hr'].append(candidate_crop)

        crops['lr'] = np.array(crops['lr'])
        crops['hr'] = np.array(crops['hr'])
        return crops

    def _apply_transform(self, img, transform_selection):
        """ Повертає та перевертає вхідне зображення відповідно до transform_selection. """

        rotate = {
            0: lambda x: x,
            1: lambda x: np.rot90(x, k=1, axes=(1, 0)),  # rotate right
            2: lambda x: np.rot90(x, k=1, axes=(0, 1)),  # rotate left
        }

        flip = {
            0: lambda x: x,
            1: lambda x: np.flip(x, 0),  # flip along horizontal axis
            2: lambda x: np.flip(x, 1),  # flip along vertical axis
        }

        rot_direction = transform_selection[0]
        flip_axis = transform_selection[1]

        img = rotate[rot_direction](img)
        img = flip[flip_axis](img)

        return img

    def _transform_batch(self, batch, transforms):
        """ Трансформує кожне зображення в наборі індивідуально. """

        t_batch = np.array(
            [self._apply_transform(img, transforms[i]) for i, img in enumerate(batch)]
        )
        return t_batch

    def get_batch(self, batch_size, idx=None, flatness=0.0):
        """
        Повертає хеш з ключами ('lr', 'hr'), що містять тренувальні набори
        зображень низького та високого розширення

        Args:
            batch_size: integer.
            flatness: float в діапазоні [0,1], порогове значення рівності ділянки.
                Визначає який рівень деталізації ділянка повинна мати. Значення 0 означає будь-яка.
        """

        if not idx:
            # randomly select one image. idx is given at validation time.
            idx = np.random.choice(range(len(self.img_list['hr'])))
        img = {}
        for res in ['lr', 'hr']:
            img_path = os.path.join(self.folders[res], self.img_list[res][idx])
            img[res] = imageio.imread(img_path) / 255.0
        batch = self._crop_imgs(img, batch_size, flatness)
        transforms = np.random.randint(0, 3, (batch_size, 2))
        batch['lr'] = self._transform_batch(batch['lr'], transforms)
        batch['hr'] = self._transform_batch(batch['hr'], transforms)

        return batch

    def get_validation_batches(self, batch_size):
        """ Повертає набір для кожного зображення в сетах валідації. """

        if self.n_validation_samples:
            batches = []
            for idx in range(self.n_validation_samples):
                batches.append(self.get_batch(batch_size, idx, flatness=0.0))
            return batches
        else:
            self.logger.error(
                'Розмір сету валідації не визначено. (не працює з сетом валідації?)'
            )
            raise ValueError(
                'Розмір сету валідації не визначено. (не працює з сетом валідації?)'
            )

    def get_validation_set(self, batch_size):
        """
        Повертає набір для кожного зображення в сеті валідації
        Розрівнює та ділить їх для Keras функції model.evaluate.
        """

        if self.n_validation_samples:
            batches = self.get_validation_batches(batch_size)
            valid_set = {'lr': [], 'hr': []}
            for batch in batches:
                for res in ('lr', 'hr'):
                    valid_set[res].extend(batch[res])
            for res in ('lr', 'hr'):
                valid_set[res] = np.array(valid_set[res])
            return valid_set
        else:
            self.logger.error(
                'Розмір сету валідації не визначено. (не працює з сетом валідації?)'
            )
            raise ValueError(
                'Розмір сету валідації не визначено. (не працює з сетом валідації?)'
            )
