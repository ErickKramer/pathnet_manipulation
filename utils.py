import numpy as np 

def normalize_images(image_array):
    return image_array.astype('float32') / 255. 

def _add_salt_and_pepper(image_array, probability=.5):
    """
    Add salt and pepper noise to the image
    """
    image = np.squeeze(image_array)
    # Generate matrix of random values
    uniform_values = np.random.rand(*image_array.shape)  
    spiced_image = image.copy()
    spiced_image = spiced_image.astype('float32')
    pepper_mask = uniform_values < probability/2. 
    spiced_image[pepper_mask] = np.min(image_array)
    salty_mask = uniform_values > (1 - probability/2.)
    spiced_image[salty_mask] = np.max(image)
    return spiced_image

def add_noise(images):
    for image_idx in range(len(images)):
        images[image_idx] = _add_salt_and_pepper(images[image_idx])
    return images