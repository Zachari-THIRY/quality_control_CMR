import os
import numpy as np

def find_non_matching_pairs(mask_paths, image_paths):
    """Takes a list of paths for masks and images, and returns the paths of each which do not have a matching pair in the other.
    Eg. If an image doesn't have a corresponding mask, then this image path will be returned. 

    Parameters
    ----------
    mask_paths: list[os.PathLike]
        The list of masks paths.
    image_paths: lsit[os.PathLike]
        The list of image paths.

    Returns
    -------
    (list, list) : missing_masks_pairs, missing_image_pairs
    """
    # Retrieving directory names
    mask_set = set([os.path.dirname(path) for path in mask_paths])
    image_set = set([os.path.dirname(path) for path in image_paths])
    # Finding unmatched ones
    missing_files = mask_set.symmetric_difference(image_set)
    # Retrieving the path of unmatched
    missing_mask_pairs = [p for p in mask_paths if os.path.dirname(p) in missing_files and "mask.nii.gz" in p]
    missing_image_pairs = [p for p in image_paths if os.path.dirname(p) in missing_files and "image.nii.gz" in p]

    return missing_mask_pairs, missing_image_pairs

def ignore_directories(mask_paths, image_paths):
    """
    Takes as input two sets of paths and removes the ones who don't have a matching one according to find_non_matching_pairs.
    
    Parameters
    ----------
    mask_paths: list[os.PathLike]
        The list of masks paths.
    image_paths: lsit[os.PathLike]
        The list of image paths.

    Returns
    -------
    np.ndarray : the list containing the directories to ignore
    """
    missing_mask_pairs, missing_image_pairs = find_non_matching_pairs(mask_paths, image_paths)
    missing_image_pairs = [os.path.dirname(p) for p in missing_image_pairs]
    missing_mask_pairs = [os.path.dirname(p) for p in missing_mask_pairs]

    ignore_directories = np.unique(missing_mask_pairs, missing_image_pairs)

    return ignore_directories
