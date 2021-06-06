import yaml
import numpy as np
from pathlib import Path

from VSR.utils.logger import get_logger
from VSR.utils.utils import get_timestamp


class TrainerHelper:
    """Колекція корисних функцій для менеджменту навчання.

    Args:
        generator: Keras модель.
        logs_dir: шлях до каталогу де логи tensorboard будуть збережені.
        weights_dir: шлях до каталогу де будуть збережені ваги.
        lr_train_dir: шлях до каталогу вхідних зображень для навчання (низьке розширення).
        feature_extractor: Keras модель VGG19.
        discriminator: Keras model, мережа дискримінатора для генеративно-змагальної мережі
        dataname: string, використовується для ідентифікації датасету під час сессії тренування.
        weights_dictionary містить шлях до вагів генератора та дискримінатора.
        fallback_save_every_n_epochs: integer, визначає через скільки епох зберігати ваги, навіть коли метрики не покращились.
        max_n_best_weights: максимальне значення вагів, які зберігаються для кращих метрик.
        max_n_other_weights: максимальне значення не кращих вагів, які зберігаються.


    Methods:
        print_training_setting.
        on_epoch_end.
        epoch_n_from_weights_name.
        initialize_training.

    """

    def __init__(
            self,
            generator,
            weights_dir,
            logs_dir,
            lr_train_dir,
            feature_extractor=None,
            discriminator=None,
            dataname=None,
            weights_generator=None,
            weights_discriminator=None,
            fallback_save_every_n_epochs=2,
            max_n_other_weights=5,
            max_n_best_weights=5,
    ):
        self.generator = generator
        self.dirs = {'logs': Path(logs_dir), 'weights': Path(weights_dir)}
        self.feature_extractor = feature_extractor
        self.discriminator = discriminator
        self.dataname = dataname

        if weights_generator:
            self.pretrained_generator_weights = Path(weights_generator)
        else:
            self.pretrained_generator_weights = None

        if weights_discriminator:
            self.pretrained_discriminator_weights = Path(weights_discriminator)
        else:
            self.pretrained_discriminator_weights = None

        self.fallback_save_every_n_epochs = fallback_save_every_n_epochs
        self.lr_dir = Path(lr_train_dir)
        self.basename = self._make_basename()
        self.session_id = self.get_session_id(basename=None)
        self.session_config_name = 'session_config.yml'
        self.callback_paths = self._make_callback_paths()
        self.weights_name = self._weights_name(self.callback_paths)
        self.best_metrics = {}
        self.since_last_epoch = 0
        self.max_n_other_weights = max_n_other_weights
        self.max_n_best_weights = max_n_best_weights
        self.logger = get_logger(__name__)

    def _make_basename(self):
        """ Комбінує назву генератора та параметрів архітектури. """

        gen_name = self.generator.name
        params = [gen_name]
        for param in np.sort(list(self.generator.params.keys())):
            params.append('{g}{p}'.format(g=param, p=self.generator.params[param]))
        return '-'.join(params)

    def get_session_id(self, basename):
        """ Повертає унікальне значення сессії. """

        time_stamp = get_timestamp()

        if basename:
            session_id = '{b}_{ts}'.format(b=basename, ts=time_stamp)
        else:
            session_id = time_stamp
        return session_id

    def _get_previous_conf(self):
        """ Перевіряє наявність конфігурації сессії session_config.yml для вагів. """

        if self.pretrained_generator_weights:
            session_config_path = (
                    self.pretrained_generator_weights.parent / self.session_config_name
            )
            if session_config_path.exists():
                return yaml.load(session_config_path.read_text(), Loader=yaml.FullLoader)
            else:
                self.logger.warning('Не вдалось знайти попередню конфігурацію')
                return {}

        return {}

    def update_config(self, training_settings):
        """
        Додає до існуючих налаштувань поточні значення.
        """

        session_settings = self._get_previous_conf()
        session_settings.update({self.session_id: training_settings})

        return session_settings

    def _make_callback_paths(self):
        """ Створює шлях, що використовуєтья для менеджменту лог файлів та вагів. """

        callback_paths = {}
        callback_paths['weights'] = self.dirs['weights'] / self.basename / self.session_id
        callback_paths['logs'] = self.dirs['logs'] / self.basename / self.session_id
        return callback_paths

    def _weights_name(self, callback_paths):
        """ Будує назву вагів тренувальної сессії. """

        w_name = {
            'generator': callback_paths['weights']
                         / (self.basename + '{metric}_epoch{epoch:03d}.hdf5')
        }
        if self.discriminator:
            w_name.update(
                {
                    'discriminator': callback_paths['weights']
                                     / (self.discriminator.name + '{metric}_epoch{epoch:03d}.hdf5')
                }
            )
        return w_name

    def print_training_setting(self, settings):
        """ Друкує налаштування навчання. """

        self.logger.info('\nДеталі навчання:')
        for k in settings[self.session_id]:
            if isinstance(settings[self.session_id][k], dict):
                self.logger.info('  {}: '.format(k))
                for kk in settings[self.session_id][k]:
                    self.logger.info(
                        '    {key}: {value}'.format(
                            key=kk, value=str(settings[self.session_id][k][kk])
                        )
                    )
            else:
                self.logger.info(
                    '  {key}: {value}'.format(key=k, value=str(settings[self.session_id][k]))
                )

    def _save_weights(self, epoch, generator, discriminator=None, metric=None, best=False):
        """ Зберігає ваги існуючих моделей. """

        if best:
            gen_path = self.weights_name['generator'].with_name(
                (self.weights_name['generator'].name).format(
                    metric='_best-' + metric, epoch=epoch + 1
                )
            )
        else:
            gen_path = self.weights_name['generator'].with_name(
                (self.weights_name['generator'].name).format(metric='', epoch=epoch + 1)
            )
        # CANT SAVE MODEL DUE TO TF LAYER INSIDE LAMBDA (PIXELSHUFFLE)
        generator.save_weights(gen_path.as_posix())
        if discriminator:
            if best:
                discr_path = self.weights_name['discriminator'].with_name(
                    (self.weights_name['discriminator'].name).format(
                        metric='_best-' + metric, epoch=epoch + 1
                    )
                )
            else:
                discr_path = self.weights_name['discriminator'].with_name(
                    (self.weights_name['discriminator'].name).format(metric='', epoch=epoch + 1)
                )
            discriminator.model.save_weights(discr_path.as_posix())
        try:
            self._remove_old_weights(self.max_n_other_weights, max_best=self.max_n_best_weights)
        except Exception as e:
            self.logger.warning('Не вдалося видалити ваги: {}'.format(e))

    def _remove_old_weights(self, max_n_weights, max_best=5):
        """
        Сканує каталог вагів та видаляє все, окрім:
            - max_best кращі нові ваги.
            - max_n_weights інші ваги.
        """

        w_list = {}
        w_list['all'] = [w for w in self.callback_paths['weights'].iterdir() if '.hdf5' in w.name]
        w_list['best'] = [w for w in w_list['all'] if 'best' in w.name]
        w_list['others'] = [w for w in w_list['all'] if w not in w_list['best']]
        # remove older best
        epochs_set = {}
        epochs_set['best'] = list(
            set([self.epoch_n_from_weights_name(w.name) for w in w_list['best']])
        )
        epochs_set['others'] = list(
            set([self.epoch_n_from_weights_name(w.name) for w in w_list['others']])
        )
        keep_max = {'best': max_best, 'others': max_n_weights}
        for type in ['others', 'best']:
            if len(epochs_set[type]) > keep_max[type]:
                epoch_list = np.sort(epochs_set[type])[::-1]
                epoch_list = epoch_list[0: keep_max[type]]
                for w in w_list[type]:
                    if self.epoch_n_from_weights_name(w.name) not in epoch_list:
                        w.unlink()

    def on_epoch_end(self, epoch, losses, generator, discriminator=None, metrics={}):
        """
        Після кожної епохи перевіряє метрики, зберігає ваги, виконує логування.
        """

        self.logger.info(losses)
        monitor_op = {'max': np.greater, 'min': np.less}
        extreme = {'max': -np.Inf, 'min': np.Inf}
        for metric in metrics:
            if metric in losses.keys():
                if metric not in self.best_metrics.keys():
                    self.best_metrics[metric] = extreme[metrics[metric]]

                if monitor_op[metrics[metric]](losses[metric], self.best_metrics[metric]):
                    self.logger.info(
                        '{} покращення від {:10.5f} до {:10.5f}'.format(
                            metric, self.best_metrics[metric], losses[metric]
                        )
                    )
                    self.logger.info('Зберігаю ваги')
                    self.best_metrics[metric] = losses[metric]
                    self._save_weights(epoch, generator, discriminator, metric=metric, best=True)
                    self.since_last_epoch = 0
                    return True
                else:
                    self.logger.info('{} не покращились.'.format(metric))
                    if self.since_last_epoch >= self.fallback_save_every_n_epochs:
                        self.logger.info('Все одно зберігаю ваги.')
                        self._save_weights(epoch, generator, discriminator, best=False)
                        self.since_last_epoch = 0
                        return True

            else:
                self.logger.warning('{} не відстежується, не можливо зберігти ваги.'.format(metric))
        self.since_last_epoch += 1
        return False

    def epoch_n_from_weights_name(self, w_name):
        """
        Витягує останнє значення епохи із назви вагів.
        """
        try:
            starting_epoch = int(w_name.split('epoch')[1][0:3])
        except Exception as e:
            self.logger.warning(
                'Не можливо отримати початкову епогу із назви вагів: \n{}'.format(w_name)
            )
            self.logger.error(e)
            starting_epoch = 0
        return starting_epoch

    def initialize_training(self, object):
        """Виконується до навчання.

        завантажує ваги, генерує назви сессій та вагів, створює каталоги та друкує дані по сессії навчання.
        """

        object.weights_generator = self.pretrained_generator_weights
        object.weights_discriminator = self.pretrained_discriminator_weights
        object._load_weights()
        w_name = object.weights_generator
        if w_name:
            last_epoch = self.epoch_n_from_weights_name(w_name.name)
        else:
            last_epoch = 0

        self.callback_paths = self._make_callback_paths()
        self.callback_paths['weights'].mkdir(parents=True)
        self.callback_paths['logs'].mkdir(parents=True)
        object.settings['training_parameters']['starting_epoch'] = last_epoch
        self.settings = self.update_config(object.settings)
        self.print_training_setting(self.settings)
        yaml.dump(
            self.settings, (self.callback_paths['weights'] / self.session_config_name).open('w')
        )
        return last_epoch
