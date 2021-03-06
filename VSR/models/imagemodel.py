import numpy as np

from VSR.utils.image_processing import (
    process_array,
    process_output,
    split_image_into_overlapping_patches,
    stich_together,
)


class ImageModel:
    """Батьківський клас моделі.

    Містить типові функції для всіх SR моделей.
    """

    def predict(self, input_image_array, by_patch_of_size=None, batch_size=10, padding_size=2):
        """
        Оброблює вхідний масив до необхідного формату та трансформує вихід мережі

        Args:
            input_image_array: масив вхідного зображення.
            by_patch_of_size: інтерфейс для великих зображень. Ділить зображення на ділянки вхідного розміру.
            padding_size: інтерфейс для великих зображень. Відступи між ділянками.
            batch_size: інтерфейс для великих зображень. Кількість оброблюваних ділянок за раз.
        Returns:
            sr_img: зображення.
        """

        if by_patch_of_size:
            lr_img = process_array(input_image_array, expand=False)
            patches, p_shape = split_image_into_overlapping_patches(
                lr_img, patch_size=by_patch_of_size, padding_size=padding_size
            )
            # повертає ділянки зображення
            for i in range(0, len(patches), batch_size):
                batch = self.model.predict(patches[i: i + batch_size])
                if i == 0:
                    collect = batch
                else:
                    collect = np.append(collect, batch, axis=0)

            scale = self.scale
            padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
            scaled_image_shape = tuple(np.multiply(input_image_array.shape[0:2], scale)) + (3,)
            sr_img = stich_together(
                collect,
                padded_image_shape=padded_size_scaled,
                target_shape=scaled_image_shape,
                padding_size=padding_size * scale,
            )

        else:
            lr_img = process_array(input_image_array)
            sr_img = self.model.predict(lr_img)[0]

        sr_img = process_output(sr_img)
        return sr_img
