from skimage.morphology import erosion, dilation
import numpy as np
import random

def __init__(kernel = 'default') -> None:
    kernel =[
    [0,1,0],
    [1,1,1],
    [0,1,0]
    ] if kernel == 'default' else kernel

def erode(sample, kernel = 'default', iterations = 1):
    kernel = [
    [0,1,0],
    [1,1,1],
    [0,1,0]
    ] if kernel == 'default' else kernel

    i = 0
    while i < iterations:
        sample = erosion(sample, kernel)
        i += 1
    return sample

def dilate(sample, kernel = 'default', iterations = 1):
    kernel = [
    [0,1,0],
    [1,1,1],
    [0,1,0]
    ] if kernel == 'default' else kernel

    i = 0
    while i < iterations:
        sample = dilation(sample, kernel)
        i += 1
    return sample

    
def remove_columns(image, hardness):
    # Ensure the hardness is valid
    if not 0 <= hardness < 1:
        raise ValueError("Hardness must be between 0 and 1")
    # Calculate the number of columns to remove
    num_columns = int(np.round(image.shape[1] * hardness))
    # Randomly choose the columns to remove
    columns_to_remove = np.random.choice(image.shape[1], num_columns, replace=False)
    # Set the chosen columns to zero
    for col in columns_to_remove:
        image[:, col] = 0
    return image


def copy_and_translate(image, hardness:float = 0.5):
    """
        This function takes as imput an image (numpy array), selects a portion of the image
        based on the given hardness, and randomly copies it somewhere else on the initial
        image.

        Parameters:
        -----------
            image:
                A Numpy array representation of the image
            hardness:float
                Between 0 and 1, 
    """

    assert hardness>0 and hardness <1, "hardness must be strictly between 0 and 1"
    
    new_image = np.copy(image)
    # Retrieving information about the image size and the portion size
    image_width, image_height = np.shape(image)
    portion_size = int(hardness*min(image_width, image_height))
    # Selecting a portion to copy
    x1 = random.randint(0, image_width-portion_size)
    y1 = random.randint(0, image_height-portion_size)
    x2, y2 = x1+portion_size, y1+portion_size

    portion = image[x1:x2, y1:y2]
    # Selecting a spot where to paste the portion
    X1 = random.randint(0, image_width - portion_size)
    Y1 = random.randint(0, image_height - portion_size)

    # Pasting the portion
    new_image[X1:X1+portion_size,Y1:Y1+portion_size] = portion

    return new_image

def alter_image(image, parameters = None):

    # Ensure the input is a valid numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array")
    # Setting default parameters
    parameters = {
        'param_copy': np.random.random()*0.7,
        'param_erode' : np.random.randint(0,20),
        'param_dilate' : np.random.randint(0,20),
        'param_columns' : np.random.random()*0.3

    } if parameters is None else parameters
    
    image = copy_and_translate(image, parameters['param_copy'])
    image = erode(image, iterations = parameters['param_erode'])
    image = dilate(image, iterations = parameters['param_dilate'])
    image = remove_columns(image, parameters['param_columns'])
    return image

