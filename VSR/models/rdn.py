import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda, UpSampling2D
from tensorflow.keras.models import Model

from VSR.models.imagemodel import ImageModel

def make_model(arch_params, patch_size):
    """ Повертає модель.

    Використовується для вибору моделі.
    """

    return RDN(arch_params, patch_size)

class RDN(ImageModel):
    """Імплементація моделі RDN для ПРЗ.

    Мережа описана в https://arxiv.org/abs/1802.08797 (Zhang et al. 2018).
    rdn = RDN(arch_params={'C': 5, 'D':16, 'G':48, 'G0':52, 'x':3})
    rdn.model.load_weights('PATH/TO/WEIGHTS')

    Args:
        arch_params: dictionary, містить параметри мережі C, D, G, G0, x.
        patch_size: integer або None, визначає вхідний розмір. Необхідна лише під час навчання.
        c_dim: integer, кількість каналів вхідного зображення.
        kernel_size: integer, звичний розмір ядра (фільтра) для згорток.
        upscaling: string, 'ups' або 'shuffle', визначає імплементацію шару збільшення.
        init_extreme_val: максимальні значення для ініціалізації RandomUniform.

    Attributes:
        C: integer, кількість шарів згорток в кожному залишковому щільному блоці (RDB).
        D: integer, кількість блоків RDB.
        G: integer, кількість вихідних фільтрів згорток в блоках RDB.
        G0: integer, кількість вихідних фільтрів кожного RDB.
        x: integer, коефіцієнт збільшення.
        model: Keras модель для RDN.
        name: назва для ідентифікації мережі збільшення під час навчання.
        model._name: ідентифікує дану мережу як мережу генератор в комплексній моделі, що створюється класом навчання.
    """

    def __init__(
            self,
            arch_params={},
            patch_size=None,
            c_dim=3,
            kernel_size=3,
            upscaling='ups',
            init_extreme_val=0.05,
            weights=''
    ):

        self.params = arch_params
        self.C = self.params['C']
        self.D = self.params['D']
        self.G = self.params['G']
        self.G0 = self.params['G0']
        self.scale = self.params['x']
        self.patch_size = patch_size
        self.c_dim = c_dim
        self.kernel_size = kernel_size
        self.upscaling = upscaling
        self.initializer = RandomUniform(
            minval=-init_extreme_val, maxval=init_extreme_val, seed=None
        )
        self.model = self._build_rdn()
        self.model._name = 'generator'
        self.name = 'rdn'

    def _upsampling_block(self, input_layer):
        """ Блок Upsampling для старих вагів. """

        x = Conv2D(
            self.c_dim * self.scale ** 2,
            kernel_size=3,
            padding='same',
            name='UPN3',
            kernel_initializer=self.initializer,
        )(input_layer)
        return UpSampling2D(size=self.scale, name='UPsample')(x)

    def _pixel_shuffle(self, input_layer):
        """ Імплементація PixelShuffle для шару підвищення. """

        x = Conv2D(
            self.c_dim * self.scale ** 2,
            kernel_size=3,
            padding='same',
            name='UPN3',
            kernel_initializer=self.initializer,
        )(input_layer)
        return Lambda(
            lambda x: tf.nn.depth_to_space(x, block_size=self.scale, data_format='NHWC'),
            name='PixelShuffle',
        )(x)

    def _UPN(self, input_layer):
        """ Шари збільшення."""

        x = Conv2D(
            64,
            kernel_size=5,
            strides=1,
            padding='same',
            name='UPN1',
            kernel_initializer=self.initializer,
        )(input_layer)
        x = Activation('relu', name='UPN1_Relu')(x)
        x = Conv2D(
            32, kernel_size=3, padding='same', name='UPN2', kernel_initializer=self.initializer
        )(x)
        x = Activation('relu', name='UPN2_Relu')(x)
        if self.upscaling == 'shuffle':
            return self._pixel_shuffle(x)
        elif self.upscaling == 'ups':
            return self._upsampling_block(x)
        else:
            raise ValueError('Невірний вибір шару збільшення роздільної здатності.')

    def _RDBs(self, input_layer):
        """RDB блоки.

        Args:
            input_layer: вхідний шар RDB блоку (другий шар згортки F_0).

        Returns:
            об'єднання шарів ознак RDB блоків із шарами ознак G0.
        """
        rdb_concat = list()
        rdb_in = input_layer
        for d in range(1, self.D + 1):
            x = rdb_in
            for c in range(1, self.C + 1):
                F_dc = Conv2D(
                    self.G,
                    kernel_size=self.kernel_size,
                    padding='same',
                    kernel_initializer=self.initializer,
                    name='F_%d_%d' % (d, c),
                )(x)
                F_dc = Activation('relu', name='F_%d_%d_Relu' % (d, c))(F_dc)
                # об'єднати вхід та вихід ConvRelu блока
                # x = [input_layer,F_11(input_layer),F_12([input_layer,F_11(input_layer)]), F_13..]
                x = concatenate([x, F_dc], axis=3, name='RDB_Concat_%d_%d' % (d, c))
            # 1x1 згортка (Локальне злиття ознак)
            x = Conv2D(
                self.G0, kernel_size=1, kernel_initializer=self.initializer, name='LFF_%d' % (d)
            )(x)
            # Локальне залишкове навчання F_{i,LF} + F_{i-1}
            rdb_in = Add(name='LRL_%d' % (d))([x, rdb_in])
            rdb_concat.append(rdb_in)

        assert len(rdb_concat) == self.D

        return concatenate(rdb_concat, axis=3, name='LRLs_Concat')

    def _build_rdn(self):
        LR_input = Input(shape=(self.patch_size, self.patch_size, 3), name='LR')
        F_m1 = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='F_m1',
        )(LR_input)
        F_0 = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='F_0',
        )(F_m1)
        FD = self._RDBs(F_0)
        # Глобальне об'єднання ознак
        # 1x1 Conv of concat RDB layers -> G0 feature maps
        GFF1 = Conv2D(
            self.G0,
            kernel_size=1,
            padding='same',
            kernel_initializer=self.initializer,
            name='GFF_1',
        )(FD)
        GFF2 = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='GFF_2',
        )(GFF1)
        # Глобальне залишкове навчання для щільних ознак
        FDF = Add(name='FDF')([GFF2, F_m1])
        # Збільшення роздільної здатності
        FU = self._UPN(FDF)
        # Збірка в зображення
        SR = Conv2D(
            self.c_dim,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='SR',
        )(FU)

        return Model(inputs=LR_input, outputs=SR)
