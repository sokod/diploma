import tensorflow.keras.backend as K


def PSNR(y_true, y_pred, MAXp=1):
    """
    Визначає значення PSNR:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).

    Args:
        y_true: еталон.
        y_pred: покращене.
        MAXp: максимальне значення пікселя (default=1).
    """
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


def RGB_to_Y(image):
    """ Зображення має значення від 0 до 1. """

    R = image[:, :, :, 0]
    G = image[:, :, :, 1]
    B = image[:, :, :, 2]
    # https://ru.wikipedia.org/wiki/YCbCr
    Y = 16 + (65.738 * R) + (129.057 * G) + (25.064 * B)
    return Y / 255.0


def PSNR_Y(y_true, y_pred, MAXp=1):
    """
    Визначає значення PSNR по каналу Y:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).

    Args:
        y_true: еталон.
        y_pred: покращене зображеня.
        MAXp: максимальне значення пікселя в діапазоні (default=1).
    """
    y_true = RGB_to_Y(y_true)
    y_pred = RGB_to_Y(y_pred)
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
