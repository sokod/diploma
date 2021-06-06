import numpy as np


def process_array(image_array, expand=True):
    """ Оброблює 3-вимірний масив в збільшену, 4 вимірну вибірку розміром 1. """

    image_batch = image_array / 255.0
    if expand:
        image_batch = np.expand_dims(image_batch, axis=0)
    return image_batch


def process_output(output_tensor):
    """ Трансформує 4-вимірний вихід тензора в зображення. """

    sr_img = output_tensor.clip(0, 1) * 255
    sr_img = np.uint8(sr_img)
    return sr_img


def pad_patch(image_patch, padding_size, channel_last=True):
    """ Додає відступ до image_patch із padding_size значенням від границь. """

    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            'edge',
        )
    else:
        return np.pad(
            image_patch,
            ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            'edge',
        )


def unpad_patches(image_patches, padding_size):
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """ Ділить зображення в частково перекриті ділянки.

    Ділянки перекриваються за padding_size значенням в пікселях.
    Args:
        image_array: масив numpy вхідного зображення.
        patch_size: розмір ділянки із оригінального зображення.
        padding_size: розмір перекритої ділянки.
    """

    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size

    # modulo here is to avoid extending of patch_size instead of 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size

    # make sure the image is divisible into regular patches
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')

    # add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)

    xmax, ymax, _ = padded_image.shape
    patches = []

    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)

    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)

    return np.array(patches), padded_image.shape


def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    """ Реконструює зображення із перекритих ділянок.

    Args:
        patches: ділянки отримані із split_image_into_overlapping_patches
        padded_image_shape: форма зображень отриманих в split_image_into_overlapping_patches
        target_shape: форма фінального зображення
        padding_size: розмір перекритої ділянки.
    """

    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size

    complete_image = np.zeros((xmax, ymax, 3))

    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
        row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size, :
        ] = patches[i]
        col += 1
    return complete_image[0: target_shape[0], 0: target_shape[1], :]
