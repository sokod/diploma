import os
import argparse
from datetime import datetime

import numpy as np
import yaml

from VSR.utils.logger import get_logger

logger = get_logger(__name__)


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', action='store_true', dest='prediction')
    parser.add_argument('--training', action='store_true', dest='training')
    parser.add_argument('--summary', action='store_true', dest='summary')
    parser.add_argument('--default', action='store_true', dest='default')
    parser.add_argument('--config', action='store', dest='config_file')
    return parser


def parse_args():
    """ Зчитуэ CLI аргументи. """

    parser = _get_parser()
    args = vars(parser.parse_args())
    if args['prediction'] and args['training']:
        logger.error('Оберіть тільки "prediction" - передбачення, чи "training" - навчання.')
        raise ValueError('Оберіть тільки "prediction" чи "training".')
    return args


def get_timestamp():
    ts = datetime.now()
    time_stamp = '{y}-{m:02d}-{d:02d}_{h:02d}{mm:02d}'.format(
        y=ts.year, m=ts.month, d=ts.day, h=ts.hour, mm=ts.minute
    )
    return time_stamp


def check_parameter_keys(parameter, needed_keys, optional_keys=None, default_value=None):
    if needed_keys:
        for key in needed_keys:
            if key not in parameter:
                logger.error('{p} не вказано ключ {k}'.format(p=parameter, k=key))
                raise
        object.settings['training_parameters']['starting_epoch'] = last_epoch
        self.settings = self.update_config(object.settings)
    if optional_keys:
        for key in optional_keys:
            if key not in parameter:
                logger.info('Встановлення {k} в {p} до {d}'.format(k=key, p=parameter, d=default_value))
                parameter[key] = default_value


def get_config_from_weights(w_path, arch_params, name):
    """
    Витягує параметри архітектури із файлу або назви вагів
    """

    w_path = os.path.basename(w_path)
    parts = w_path.split(name)[1]
    parts = parts.split('_')[0]
    parts = parts.split('-')
    new_param = {}
    for param in arch_params:
        param_part = [x for x in parts if param in x]
        param_value = int(param_part[0].split(param)[1])
        new_param[param] = param_value
    return new_param


def select_option(options, message='', val=None):
    """ вибір наданої CLI опцій. """

    while val not in options:
        val = input(message)
        if val not in options:
            logger.error('Невірний вибір.')
    return val


def select_multiple_options(options, message='', val=None):
    """ Вибір декілької CLI опцій. """

    n_options = len(options)
    valid_selections = False
    selected_options = []
    while not valid_selections:
        for i, opt in enumerate(np.sort(options)):
            logger.info('{}: {}'.format(i, opt))
        val = input(message + ' (вибірка поділена пробілом)\n')
        vals = val.split(' ')
        valid_selections = True
        for v in vals:
            if int(v) not in list(range(n_options)):
                logger.error('Невірний вибір.')
                valid_selections = False
            else:
                selected_options.append(options[int(v)])

    return selected_options


def select_bool(message=''):
    """ CLI зчитування логічної змінної. """

    options = ['т', 'н']
    message = message + ' (' + '/'.join(options) + ') '
    val = None
    while val not in options:
        val = input(message)
        if val not in options:
            logger.error('Введіть т (так) чи н (ні).')
    if val == 'т':
        return True
    elif val == 'н':
        return False


def select_positive_float(message=''):
    """ CLI не негативне значення float. """

    value = -1
    while value < 0:
        value = float(input(message))
        if value < 0:
            logger.error('Невірний вибір.')
    return value


def select_positive_integer(message='', value=-1):
    """ CLI не негативне значення int. """

    while value < 0:
        value = int(input(message))
        if value < 0:
            logger.error('Невірний вибір.')
    return value


def browse_weights(weights_dir, model='generator'):
    """ Вибірка вагів із CLI. """

    exit = False
    while exit is False:
        weights = np.sort(os.listdir(weights_dir))[::-1]
        print_sel = dict(zip(np.arange(len(weights)), weights))
        for k in print_sel.keys():
            logger_message = '{item_n}: {item} \n'.format(item_n=k, item=print_sel[k])
            logger.info(logger_message)

        sel = select_positive_integer('>>> Оберіть каталог чи ваги для {}\n'.format(model))
        if weights[sel].endswith('hdf5'):
            weights_path = os.path.join(weights_dir, weights[sel])
            exit = True
        else:
            weights_dir = os.path.join(weights_dir, weights[sel])
    return weights_path


def setup(config_file='config.yml', default=False, training=False, prediction=False):
    """Інтерфейс CLI для встановлення навчання чи передбачення (класифікації).

    Зчитує шлях до конфігурації та аргументів із CLI.
    """

    conf = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    if training:
        session_type = 'training'
    elif prediction:
        session_type = 'prediction'
    else:
        message = '(н)авчання - навчання чи (п)ередбачення - передбачення? (н/п) '
        session_type = {'н': 'training', 'п': 'prediction'}[select_option(['н', 'п'], message)]
    if default:
        all_default = 'y'
    else:
        all_default = select_bool('Значення за замовчуванням для всього?')

    if all_default:
        generator = conf['default']['generator']
        if session_type == 'prediction':
            dataset = conf['default']['test_set']
            conf['generators'][generator] = get_config_from_weights(
                conf['weights_paths']['generator'], conf['generators'][generator], generator
            )
        elif session_type == 'training':
            dataset = conf['default']['training_set']

        return session_type, generator, conf, dataset

    logger.info('Оберіть мережу покращення')
    generators = {}
    for i, gen in enumerate(conf['generators']):
        generators[str(i)] = gen
        logger.info('{}: {}'.format(i, gen))
    generator = generators[select_option(generators)]

    load_weights = input('Завантажити існуючі ваги для {}? ([т]/н/з) '.format(generator))
    if load_weights == 'н':
        default = select_bool('Завантажити параметри за замовчуванням для {}?'.format(generator))
        if not default:
            for param in conf['generators'][generator]:
                value = select_positive_integer(message='{}:'.format(param))
                conf['generators'][generator][param] = value
        else:
            logger.info('Параметри за замовчуванням {}.'.format(generator))
    elif (load_weights == 'з') and (conf['weights_paths']['generator']):
        logger.info('Завантаження вагів за замовчуванням для {}'.format(generator))
        logger.info(conf['weights_paths']['generator'])
        conf['generators'][generator] = get_config_from_weights(
            conf['weights_paths']['generator'], conf['generators'][generator], generator
        )
    else:
        conf['weights_paths']['generator'] = browse_weights(conf['dirs']['weights'], generator)
        conf['generators']['generator'] = get_config_from_weights(
            conf['weights_paths']['generator'], conf['generators'][generator], generator
        )
    logger.info('{} параметри:'.format(generator))
    logger.info(conf['generators'][generator])

    if session_type == 'training':
        default_loss_weights = select_bool('Використовувати ваги за замовчуванням для компоненту втрат (loss)?')
        if not default_loss_weights:
            conf['loss_weights']['generator'] = select_positive_float(
                'Введіть коефіцієнт генератора попіксельного компонента втрат '
            )
        use_discr = select_bool('Використовувати конкурентну мережу?')
        if use_discr:
            conf['default']['discriminator'] = True
            discr_w = select_bool('Використовувати існуючі ваги дескримінатора?')
            if discr_w:
                conf['weights_paths']['discriminator'] = browse_weights(
                    conf['dirs']['weights'], 'discriminator'
                )
            if not default_loss_weights:
                conf['loss_weights']['discriminator'] = select_positive_float(
                    'Введіть коефіцієнт для компоненту втрат конкуретної мережі '
                )

        use_feature_extractor = select_bool('Використосувати feature extractor?')
        if use_feature_extractor:
            conf['default']['feature_extractor'] = True
            if not default_loss_weights:
                conf['loss_weights']['feature_extractor'] = select_positive_float(
                    'Введіть коефіцієнт для функції втрат компонента conv '
                )
        default_metrics = select_bool('Моніторити стандартні показники?')
        if not default_metrics:
            suggested_list = suggest_metrics(use_discr, use_feature_extractor)
            selected_metrics = select_multiple_options(
                list(suggested_list.keys()), message='Оберіть показники для моніторингу.'
            )

            conf['session']['training']['monitored_metrics'] = {}
            for metric in selected_metrics:
                conf['session']['training']['monitored_metrics'][metric] = suggested_list[metric]
            print(conf['session']['training']['monitored_metrics'])

    dataset = select_dataset(session_type, conf)

    return session_type, generator, conf, dataset


def suggest_metrics(discriminator=False, feature_extractor=False, loss_weights={}):
    suggested_metrics = {}
    if not discriminator and not feature_extractor:
        suggested_metrics['val_loss'] = 'min'
        suggested_metrics['train_loss'] = 'min'
        suggested_metrics['val_PSNR'] = 'max'
        suggested_metrics['train_PSNR'] = 'max'
    if feature_extractor or discriminator:
        suggested_metrics['val_generator_loss'] = 'min'
        suggested_metrics['train_generator_loss'] = 'min'
        suggested_metrics['val_generator_PSNR'] = 'max'
        suggested_metrics['train_generator_PSNR'] = 'max'
    if feature_extractor:
        suggested_metrics['val_feature_extractor_loss'] = 'min'
        suggested_metrics['train_feature_extractor_loss'] = 'min'
    return suggested_metrics


def select_dataset(session_type, conf):
    """ вибір датасету для навчання через CLI. """

    if session_type == 'training':
        logger.info('Оберіть набір для навчання')
        datasets = {}
        for i, data in enumerate(conf['training_sets']):
            datasets[str(i)] = data
            logger.info('{}: {}'.format(i, data))
        dataset = datasets[select_option(datasets)]

        return dataset
    else:
        logger.info('Оберіть тестовий набір')
        datasets = {}
        for i, data in enumerate(conf['test_sets']):
            datasets[str(i)] = data
            logger.info('{}: {}'.format(i, data))
        dataset = datasets[select_option(datasets)]

        return dataset
