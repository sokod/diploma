from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

from VSR.utils.logger import get_logger


class Cut_VGG19:
    """
    Клас що отримує мережу VGG19 з Keras, навчану на датасеті imagenet, та визначає
    <layers_to_extract> як вихідні шари.

    Args:
        layers_to_extract: список вихідних шарів.
        patch_size: integer, визначає розмір входу (patch_size x patch_size).

    Attributes:
        loss_model: vgg архітектура з <layers_to_extract>, як вихідні шари.
    """

    def __init__(self, patch_size, layers_to_extract):
        self.patch_size = patch_size
        self.input_shape = (patch_size,) * 2 + (3,)
        self.layers_to_extract = layers_to_extract
        self.logger = get_logger(__name__)

        if len(self.layers_to_extract) > 0:
            self._cut_vgg()
        else:
            self.logger.error('Невірна ініціалізація VGG: витягнуті шари повинні бути > 0')
            raise ValueError('Невірна ініціалізація VGG: витягнуті шари повинні бути > 0')

    def _cut_vgg(self):
        """
        Завантажує навчану VGG, вихідні шари - self.layers_to_extract.
        """

        vgg = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape)
        vgg.trainable = False
        outputs = [vgg.layers[i].output for i in self.layers_to_extract]
        self.model = Model([vgg.input], outputs)
        self.model._name = 'feature_extractor'
        self.name = 'vgg19'  # використовується в назві вагів
