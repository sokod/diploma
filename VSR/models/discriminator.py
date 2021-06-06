from tensorflow.keras.layers import Input, Activation, Dense, Conv2D, BatchNormalization, \
    LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Discriminator:
    """
    Імплементація мережі дискримінатора для генеративно-змагального компоненту функції втрат

    Args:
        patch_size: integer, визначає вхідний розмір як (patch_size, patch_size, 3).
        kernel_size: розмір ядра (фільтра) блоків згортки.

    Attributes:
        model: Keras модель.
        name: назва для класифікації дискримінатора під час навчання GAN.
        model._name: ідентифікує дану мережу як дискримінатор в комплексній моделі створеній класом навчання.
        block_param: dictionary, визначає кількість фільтрів та кроків для кожного блоку згортки.

    """

    def __init__(self, patch_size, kernel_size=3):
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.block_param = {}
        self.block_param['filters'] = (64, 128, 128, 256, 256, 512, 512)
        self.block_param['strides'] = (2, 1, 2, 1, 1, 1, 1)
        self.block_num = len(self.block_param['filters'])
        self.model = self._build_disciminator()
        optimizer = Adam(0.0002, 0.5)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model._name = 'discriminator'
        self.name = 'srgan-large'

    def _conv_block(self, input, filters, strides, batch_norm=True, count=None):
        """ Шар згортки + Leaky ReLU + умовний BN. """

        x = Conv2D(
            filters,
            kernel_size=self.kernel_size,
            strides=strides,
            padding='same',
            name='Conv_{}'.format(count),
        )(input)
        x = LeakyReLU(alpha=0.2)(x)
        if batch_norm:
            x = BatchNormalization(momentum=0.8)(x)
        return x

    def _build_disciminator(self):
        """ Об'єднує шари дискримінатора. """

        HR = Input(shape=(self.patch_size, self.patch_size, 3))
        x = self._conv_block(HR, filters=64, strides=1, batch_norm=False, count=1)
        for i in range(self.block_num):
            x = self._conv_block(
                x,
                filters=self.block_param['filters'][i],
                strides=self.block_param['strides'][i],
                count=i + 2,
            )
        x = Dense(self.block_param['filters'][-1] * 2, name='Dense_1024')(x)
        x = LeakyReLU(alpha=0.2)(x)
        # x = Flatten()(x)
        x = Dense(1, name='Dense_last')(x)
        HR_v_SR = Activation('sigmoid')(x)

        discriminator = Model(inputs=HR, outputs=HR_v_SR)
        return discriminator
