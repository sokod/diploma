from time import time

import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K

from VSR.utils.datahandler import DataHandler
from VSR.utils.train_helper import TrainerHelper
from VSR.utils.metrics import PSNR
from VSR.utils.metrics import PSNR_Y
from VSR.utils.logger import get_logger
from VSR.utils.utils import check_parameter_keys


class Trainer:
    """Клас для налаштування та проведення навчання.

    Приймає на вхід генератор(модель) та створює SR зображення

    Args:
        generator: Keras модель.
        discriminator: Keras model, мережа дискримінатора для генеративно-змагальної мережі
        feature_extractor: Keras модель VGG19.
        lr_train_dir: шлях до каталогу вхідних зображень для навчання (низьке розширення).
        hr_train_dir: шлях до каталогу вхідних зображень для навчання (високе розширення).
        lr_valid_dir: шлях до каталогу вхідних зображень для валідації (низьке розширення).
        hr_valid_dir: шлях до каталогу вхідних зображень для валідації (високе розширення).
        learning_rate: float.
        loss_weights: dictionary, використовується як ваги до компонентів функцій втрат.
            Містить 'generator' для компоненту втрат генератора, може містити 'discriminator' та 'feature_extractor'
            для дискримінатора та VGG19.
        logs_dir: шлях до каталогу де логи tensorboard будуть збережені.
        weights_dir: шлях до каталогу де будуть збережені ваги.
        dataname: string, використовується для ідентифікації датасету під час сессії тренування.
        weights_generator: шлях до вагів генератора, для продовження навчання.
        weights_discriminator: шлях до вагів дискримінатора, для продовження навчання.
        n_validation:integer, кількість прикладів валідації, що використовується під час навчання.
        flatness: dictionary. Визначає поріг рівності для тренувальних ділянок.
        lr_decay_frequency: integer, кількість епох після яких зменшується коефіцієнт навчання.
        lr_decay_factor: 0 < float <1, коефіцієнт зменшення рівня навчання.

    Methods:
        train: поєднує мережі та починає навчання із встановленими параметрами.

    """

    def __init__(
            self,
            generator,
            discriminator,
            feature_extractor,
            lr_train_dir,
            hr_train_dir,
            lr_valid_dir,
            hr_valid_dir,
            loss_weights={'generator': 1.0, 'discriminator': 0.003, 'feature_extractor': 1 / 12},
            log_dirs={'logs': 'logs', 'weights': 'weights'},
            fallback_save_every_n_epochs=2,
            dataname=None,
            weights_generator=None,
            weights_discriminator=None,
            n_validation=None,
            flatness={'min': 0.0, 'increase_frequency': None, 'increase': 0.0, 'max': 0.0},
            learning_rate={'initial_value': 0.0004, 'decay_frequency': 100, 'decay_factor': 0.5},
            adam_optimizer={'beta1': 0.9, 'beta2': 0.999, 'epsilon': None},
            losses={
                'generator': 'mae',
                'discriminator': 'binary_crossentropy',
                'feature_extractor': 'mse',
            },
            metrics={'generator': 'PSNR_Y'},
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.scale = generator.scale
        self.lr_patch_size = generator.patch_size
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.weights_generator = weights_generator
        self.weights_discriminator = weights_discriminator
        self.adam_optimizer = adam_optimizer
        self.dataname = dataname
        self.flatness = flatness
        self.n_validation = n_validation
        self.losses = losses
        self.log_dirs = log_dirs
        self.metrics = metrics
        if self.metrics['generator'] == 'PSNR_Y':
            self.metrics['generator'] = PSNR_Y
        elif self.metrics['generator'] == 'PSNR':
            self.metrics['generator'] = PSNR
        self._parameters_sanity_check()
        self.model = self._combine_networks()

        self.settings = {}
        self.settings['training_parameters'] = locals()
        self.settings['training_parameters']['lr_patch_size'] = self.lr_patch_size
        self.settings = self.update_training_config(self.settings)

        self.logger = get_logger(__name__)

        self.helper = TrainerHelper(
            generator=self.generator,
            weights_dir=log_dirs['weights'],
            logs_dir=log_dirs['logs'],
            lr_train_dir=lr_train_dir,
            feature_extractor=self.feature_extractor,
            discriminator=self.discriminator,
            dataname=dataname,
            weights_generator=self.weights_generator,
            weights_discriminator=self.weights_discriminator,
            fallback_save_every_n_epochs=fallback_save_every_n_epochs,
        )

        self.train_dh = DataHandler(
            lr_dir=lr_train_dir,
            hr_dir=hr_train_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=None,
        )
        self.valid_dh = DataHandler(
            lr_dir=lr_valid_dir,
            hr_dir=hr_valid_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=n_validation,
        )

    def _parameters_sanity_check(self):
        """ Parameteres sanity check. """

        if self.discriminator:
            assert self.lr_patch_size * self.scale == self.discriminator.patch_size
            self.adam_optimizer
        if self.feature_extractor:
            assert self.lr_patch_size * self.scale == self.feature_extractor.patch_size

        check_parameter_keys(
            self.learning_rate,
            needed_keys=['initial_value'],
            optional_keys=['decay_factor', 'decay_frequency'],
            default_value=None,
        )
        check_parameter_keys(
            self.flatness,
            needed_keys=[],
            optional_keys=['min', 'increase_frequency', 'increase', 'max'],
            default_value=0.0,
        )
        check_parameter_keys(
            self.adam_optimizer,
            needed_keys=['beta1', 'beta2'],
            optional_keys=['epsilon'],
            default_value=None,
        )
        check_parameter_keys(self.log_dirs, needed_keys=['logs', 'weights'])

    def _combine_networks(self):
        """
        Створює комбіновану модель, що містить мережу генератора, дискримінатора,
        VGG19, якщо вони є.
        """

        lr = Input(shape=(self.lr_patch_size,) * 2 + (3,))
        sr = self.generator.model(lr)
        outputs = [sr]
        losses = [self.losses['generator']]
        loss_weights = [self.loss_weights['generator']]

        if self.discriminator:
            self.discriminator.model.trainable = False
            validity = self.discriminator.model(sr)
            outputs.append(validity)
            losses.append(self.losses['discriminator'])
            loss_weights.append(self.loss_weights['discriminator'])
        if self.feature_extractor:
            self.feature_extractor.model.trainable = False
            sr_feats = self.feature_extractor.model(sr)
            outputs.extend([*sr_feats])
            losses.extend([self.losses['feature_extractor']] * len(sr_feats))
            loss_weights.extend(
                [self.loss_weights['feature_extractor'] / len(sr_feats)] * len(sr_feats)
            )
        combined = Model(inputs=lr, outputs=outputs)
        # https://stackoverflow.com/questions/42327543/adam-optimizer-goes-haywire-after-200k-batches-training-loss-grows
        optimizer = Adam(
            beta_1=self.adam_optimizer['beta1'],
            beta_2=self.adam_optimizer['beta2'],
            lr=self.learning_rate['initial_value'],
            epsilon=self.adam_optimizer['epsilon'],
        )
        combined.compile(
            loss=losses, loss_weights=loss_weights, optimizer=optimizer, metrics=self.metrics
        )
        return combined

    def _lr_scheduler(self, epoch):
        """ Планувальник оновлень коефіцієнту навчання. """

        n_decays = epoch // self.learning_rate['decay_frequency']
        lr = self.learning_rate['initial_value'] * (self.learning_rate['decay_factor'] ** n_decays)
        # no lr below minimum control 10e-7
        return max(1e-7, lr)

    def _flatness_scheduler(self, epoch):
        if self.flatness['increase']:
            n_increases = epoch // self.flatness['increase_frequency']
        else:
            return self.flatness['min']

        f = self.flatness['min'] + n_increases * self.flatness['increase']

        return min(self.flatness['max'], f)

    def _load_weights(self):
        """
        Завантажує ваги із шляху
        """

        if self.weights_generator:
            self.model.get_layer('generator').load_weights(str(self.weights_generator))

        if self.discriminator:
            if self.weights_discriminator:
                self.model.get_layer('discriminator').load_weights(str(self.weights_discriminator))
                self.discriminator.model.load_weights(str(self.weights_discriminator))

    def _format_losses(self, prefix, losses, model_metrics):
        """ Створює хеш для відслідковування в tensorboard. """

        return dict(zip([prefix + m for m in model_metrics], losses))

    def update_training_config(self, settings):
        """ Генералізує налаштування навчання. """

        _ = settings['training_parameters'].pop('weights_generator')
        _ = settings['training_parameters'].pop('self')
        _ = settings['training_parameters'].pop('generator')
        _ = settings['training_parameters'].pop('discriminator')
        _ = settings['training_parameters'].pop('feature_extractor')
        settings['generator'] = {}
        settings['generator']['name'] = self.generator.name
        settings['generator']['parameters'] = self.generator.params
        settings['generator']['weights_generator'] = self.weights_generator

        _ = settings['training_parameters'].pop('weights_discriminator')
        if self.discriminator:
            settings['discriminator'] = {}
            settings['discriminator']['name'] = self.discriminator.name
            settings['discriminator']['weights_discriminator'] = self.weights_discriminator
        else:
            settings['discriminator'] = None

        if self.discriminator:
            settings['feature_extractor'] = {}
            settings['feature_extractor']['name'] = self.feature_extractor.name
            settings['feature_extractor']['layers'] = self.feature_extractor.layers_to_extract
        else:
            settings['feature_extractor'] = None

        return settings

    def train(self, epochs, steps_per_epoch, batch_size, monitored_metrics):
        """
        Підтримує навчання для заданої кількості епох, надсилає втрати до Tensorboard

        Args:
            epochs: кількість епох для тренування.
            steps_per_epoch: кількість кроків за епоху.
            batch_size: кількість зображень на крок епохи.
            monitored_metrics: dictionary, ключі - це метрики для відслідковування вагів та їх зберігання.
            Ключі - режим метрик для зберігання вагів ('min' або 'max').
        """

        self.settings['training_parameters']['steps_per_epoch'] = steps_per_epoch
        self.settings['training_parameters']['batch_size'] = batch_size
        starting_epoch = self.helper.initialize_training(
            self
        )  # load_weights, creates folders, creates basename

        self.tensorboard = TensorBoard(log_dir=str(self.helper.callback_paths['logs']))
        self.tensorboard.set_model(self.model)

        # validation data
        validation_set = self.valid_dh.get_validation_set(batch_size)
        y_validation = [validation_set['hr']]
        if self.discriminator:
            discr_out_shape = list(self.discriminator.model.outputs[0].shape)[1:4]
            valid = np.ones([batch_size] + discr_out_shape)
            fake = np.zeros([batch_size] + discr_out_shape)
            validation_valid = np.ones([len(validation_set['hr'])] + discr_out_shape)
            y_validation.append(validation_valid)
        if self.feature_extractor:
            validation_feats = self.feature_extractor.model.predict(validation_set['hr'])
            y_validation.extend([*validation_feats])

        for epoch in range(starting_epoch, epochs):
            self.logger.info('Епоха {e}/{tot_eps}'.format(e=epoch, tot_eps=epochs))
            K.set_value(self.model.optimizer.lr, self._lr_scheduler(epoch=epoch))
            self.logger.info('Поточний рівень навчання: {}'.format(K.eval(self.model.optimizer.lr)))

            flatness = self._flatness_scheduler(epoch)
            if flatness:
                self.logger.info('Поточний поріг рівності: {}'.format(flatness))

            epoch_start = time()
            for step in tqdm(range(steps_per_epoch)):
                batch = self.train_dh.get_batch(batch_size, flatness=flatness)
                y_train = [batch['hr']]
                training_losses = {}

                ## Discriminator training
                if self.discriminator:
                    sr = self.generator.model.predict(batch['lr'])
                    d_loss_real = self.discriminator.model.train_on_batch(batch['hr'], valid)
                    d_loss_fake = self.discriminator.model.train_on_batch(sr, fake)
                    d_loss_fake = self._format_losses(
                        'train_d_fake_', d_loss_fake, self.discriminator.model.metrics_names
                    )
                    d_loss_real = self._format_losses(
                        'train_d_real_', d_loss_real, self.discriminator.model.metrics_names
                    )
                    training_losses.update(d_loss_real)
                    training_losses.update(d_loss_fake)
                    y_train.append(valid)

                ## Generator training
                if self.feature_extractor:
                    hr_feats = self.feature_extractor.model.predict(batch['hr'])
                    y_train.extend([*hr_feats])

                model_losses = self.model.train_on_batch(batch['lr'], y_train)
                model_losses = self._format_losses('train_', model_losses, self.model.metrics_names)
                training_losses.update(model_losses)

                self.tensorboard.on_epoch_end(epoch * steps_per_epoch + step, training_losses)
                self.logger.debug('Похибки на кроці {s}:\n {l}'.format(s=step, l=training_losses))

            elapsed_time = time() - epoch_start
            self.logger.info('Епоха {} заняла {:10.1f}s'.format(epoch, elapsed_time))

            validation_losses = self.model.evaluate(
                validation_set['lr'], y_validation, batch_size=batch_size
            )
            validation_losses = self._format_losses(
                'val_', validation_losses, self.model.metrics_names
            )

            if epoch == starting_epoch:
                remove_metrics = []
                for metric in monitored_metrics:
                    if (metric not in training_losses) and (metric not in validation_losses):
                        msg = ' '.join([metric, 'не є серед метрик моделі, видаляю.'])
                        self.logger.error(msg)
                        remove_metrics.append(metric)
                for metric in remove_metrics:
                    _ = monitored_metrics.pop(metric)

            # should average train metrics
            end_losses = {}
            end_losses.update(validation_losses)
            end_losses.update(training_losses)

            self.helper.on_epoch_end(
                epoch=epoch,
                losses=end_losses,
                generator=self.model.get_layer('generator'),
                discriminator=self.discriminator,
                metrics=monitored_metrics,
            )
            self.tensorboard.on_epoch_end(epoch, validation_losses)
        self.tensorboard.on_train_end(None)
