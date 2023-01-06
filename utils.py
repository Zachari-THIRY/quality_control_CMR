import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import torch
import torchvision
import shutil

from typing import Callable
from IPython.display import display
from PIL import Image
from medpy.metric import binary
from scipy import stats
from scipy.ndimage import binary_fill_holes
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.augmentations.spatial_transformations import augment_spatial

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################
##Data preparation##
####################

#TODO Solve the empty array concatention issue (eg. run on data/heart/testing)

def find_segmentations(root_dir:str, keywords: list, absolute: bool = False) -> list :
    """
    Returns a list of all the paths to segmentations matching the keywords list. 
    If absolute is True, then absolute paths are returned. Otherwise, relative paths are given.

    Parameters
    ----------
        root_dir: str
            The path in which to search for segmentations
        keywords: list
            The list of strings to search for a match
        absolute: bool
            If True, absolute paths are returned
    """
    assert type(keywords) != str, "Parameter keywords must be a list of str."
    segmentations = [[]]
    cwd = os.getcwd() if absolute else ""
    for dirElement in os.listdir(root_dir):
        subPath = os.path.join(root_dir, dirElement)
        if os.path.isdir(subPath):
            segmentations.append(find_segmentations(subPath, keywords, absolute))
        else :
            for keyword in keywords:
                if keyword in dirElement:
                    path = os.path.join(cwd,root_dir, dirElement)
                    segmentations.append(path)
        
    return np.unique(np.hstack(segmentations))

#TODO Implement

def structure_dataset(segmentation_paths:list, destination_folder:str, fileName: str="mask.nii.gz", delete: str=None) -> None:

    """
    The purpose of this method is to uniformize the dataset so that all other functions work on the same directory architecture.
    All segmentations pointed by segmentation_paths will be moved to destination_folder/patientXXX/fileName

    Parameters
    ----------
        segmentation_paths: list
            A list of all the paths of the segmentations to be restructured, length must be <= 1000
        destination_folder: str
            The new root_folder of the dataset. Both relative and absolute paths are accepted
        fileName: str
            The name of the new segmentation files
        delete: str
            If set to a string path, then specified folder will be deleted. Default: None.

    Note
    ----
        This limit of 1000 is solely dued to how the names are formatted (here :03d ; see source code). \n
        The limit can be removed if the patient folder names are adjusted throughout the code.

    """

    assert fileName.endswith(".nii.gz"), "fileName must end with .nii.gz"
    assert(len(segmentation_paths) <=1000 ), "Dataset is too large. Make sure N <= 1000"

    os.makedirs(destination_folder) if not os.path.exists(destination_folder) else None
    
    segmentation_paths.sort()
    curPath = os.path.join(os.getcwd(), destination_folder)
    os.makedirs(curPath) if not os.path.exists(curPath) else None
    for i,segmentation_path in enumerate(segmentation_paths):
        target_folder = os.path.join(curPath, "patient{:03d}".format(i))
        os.makedirs(target_folder) if not os.path.exists(target_folder) else None
        target_name = os.path.join(target_folder, fileName)
        os.rename(segmentation_path, target_name)

    if delete is not None:
        try:
            shutil.rmtree(delete, ignore_errors=False)
        except:
            print("Could not delete specified folder")

#TODO : deprecate structure_dataset_name

def structure_dataset_names(REL_DATA_LOCATION, newLocation = "structured"):
    """
    Deprecated, use structure_dataset instead
    """
    # TODO: Also remove files which don't have the right masks or images ; and make sure README is moved
    DATA_LOCATION = os.path.join(os.getcwd(), REL_DATA_LOCATION)                    # Absolute path to location
    NEW_DATA_LOCATION = os.path.join(os.path.dirname(DATA_LOCATION), newLocation)   # Path of new transformed data
    os.rename(os.path.join(DATA_LOCATION, "README"), os.path.join(os.path.dirname(DATA_LOCATION), "README"))
    patients = sorted(os.listdir(DATA_LOCATION))
    assert(len(patients) <=1000 ), "Dataset is too large. Make sure N <= 1000"

    if not os.path.isdir(NEW_DATA_LOCATION):
        os.mkdir(NEW_DATA_LOCATION)

    for id, patient in enumerate(patients):
        current_name = os.path.join(DATA_LOCATION, patient)
        new_name = os.path.join(NEW_DATA_LOCATION, "patient{:03d}".format(id))
        os.rename(current_name, new_name)

    os.rmdir(DATA_LOCATION)


def crop_image(image):
    """
    Returns the sections of the image containing non zero values ; i.e the smallest 3D box containing actual segmentation.

    Parameters
    ----------
    image: array
        The three dimensional matrix representation of the segmentation

    """
    nonzero_mask = binary_fill_holes(image != 0)
    mask_voxel_coords = np.stack(np.where(nonzero_mask))
    minidx = np.min(mask_voxel_coords, axis=1)
    maxidx = np.max(mask_voxel_coords, axis=1) + 1
    resizer = tuple([slice(*i) for i in zip(minidx,maxidx)])
    return resizer

def generate_patient_info(folder, patient_ids):
    num_patients = sum(os.path.isdir(folder + i) for i in os.listdir(folder))
    patient_info = {}
    for id in patient_ids:
        patient_folder = os.path.join(folder, 'patient{:03d}'.format(id))
        patient_info[id] = {
            k:v for k,v in np.loadtxt(
                os.path.join(patient_folder, "Info.cfg"),
                dtype=str, delimiter=':'
            )
        }
        patient_info[id]["ED"] = int(patient_info[id]["ED"])
        patient_info[id]["ES"] = int(patient_info[id]["ES"])

        image = nib.load(os.path.join(patient_folder, "patient{:03d}_frame{:02d}_gt.nii.gz".format(id, patient_info[id]["ED"])))
        patient_info[id]["shape_ED"] = image.get_fdata().shape
        patient_info[id]["crop_ED"] = crop_image(image.get_fdata())
        
        image = nib.load(os.path.join(patient_folder, "patient{:03d}_frame{:02d}_gt.nii.gz".format(id, patient_info[id]["ES"])))
        patient_info[id]["shape_ES"] = image.get_fdata().shape   
        patient_info[id]["crop_ES"] = crop_image(image.get_fdata())

        patient_info[id]["spacing"] = image.header["pixdim"][[3,2,1]]
        patient_info[id]["header"] = image.header
        patient_info[id]["affine"] = image.affine
    return patient_info

def generate_patient_info_brain(folder: str, fileName: str="mask.nii.gz", verbose: bool = False, verbose_rate: int = 10):
    """
    Generates patient info from the structructured dataset in folder, based on the fileName files.\n

    Parameters
    ----------
        folder: str
            The root folder containing the structured dataset
        fileName: str
            The generic filename of each mask ; it should be the same for every patient
        verbose: bool
            Enables the progress display every verbose_rate units processed
        verbose_rate: int
            The step with which progress is displayed ; by default every 10 units
    
    Gathered information :
    ----------------------
        Shape of the original image: "shape",
        The resizer used for cropping: "crop",
        The original spacing: "spacing",
        The original image header: "header",
        The image affine: "affine".
    """

    assert verbose_rate > 0, f"The verbose rate should be positivie, but verbose_rate = {verbose_rate} was passed."

    patients_list  =sorted(os.listdir(folder))
    if patients_list[-1] == "patient_info.npy" : patients_list.pop(-1)
    patient_ids = [i for i in range(len(patients_list))]
    patient_ids.remove(9) # removing absent images or masks
    patient_info = {}
    for id in patient_ids:
        patient_folder = os.path.join(folder, 'patient{:03d}'.format(id))
        image = nib.load(os.path.join(patient_folder, fileName))
        patient_info[id] = {} # Initialising it
        patient_info[id]["shape"] = image.get_fdata().shape
        patient_info[id]["crop"] = crop_image(image.get_fdata())

        patient_info[id]["spacing"] = image.header["pixdim"][[3,2,1]]
        patient_info[id]["header"] = image.header
        patient_info[id]["affine"] = image.affine
        if(id%verbose_rate == 0 and verbose) : 
            print("Just processed patient {PATIENT_ID:03d} out of {TOTAL:03d}".format(PATIENT_ID = id, TOTAL=len(patient_ids)))
    return patient_info

def median_spacing_target(folder: str, round=2) -> list:
    """
    Returns a spacing target as the median of all patients' spacing, rounded to specified decimal.

    Parameters
    ----------
        folder: str
            The directory containing patient_info.npy
        round: int
            The number of decimals of the rounding
    """


    path = os.path.join(folder, "patient_info.npy")
    patient_info = np.load(path, allow_pickle=True).item()
    spacing_target = np.median(np.array([patient_info[key]["spacing"] for key in patient_info.keys()]), axis=0)
    spacing_target[0] = 1
    return np.round(spacing_target, round)

def preprocess_image(image, crop, spacing, spacing_target):
    """
    Returns a cropped and resized version of the image according to the specified attributes

    Parameters
    ----------
        image: array
            The three dimensional matrix representation of the image
        crop: list
            The resizer, usually the output of crop_image()
        spacing: list
            The current image spacing
        spacing_target:
            The target image spacing
    """
    # Removing the unused time dimension
    if len(image.shape) == 4: image = image[:, :, :, 0] 
    if len(crop) == 4: crop = crop[:3] 

    image = image[crop].transpose(2,1,0)
    spacing_target[0] = spacing[0]
    new_shape = np.round(spacing / spacing_target * image.shape).astype(int)
    image = resize_segmentation(image, new_shape, order=1)
    return image, new_shape

def preprocess(patient_ids, patient_info, spacing_target, folder, folder_out, get_patient_folder, get_fname):
    for id in patient_ids:
        patient_folder = get_patient_folder(folder, id)
        images = []
        for phase in ["ED","ES"]:
            fname = get_fname(patient_info, id, phase)
            fname = os.path.join(patient_folder, fname)
            if(not os.path.isfile(fname)):
                continue
            image = preprocess_image(
                nib.load(fname).get_fdata(),
                patient_info[id]["crop_{}".format(phase)],
                patient_info[id]["spacing"],
                spacing_target
            )
            images.append(image)
        images = np.vstack(images)
        np.save(os.path.join(folder_out, "patient{:03d}".format(id)), images.astype(np.float32))

def preprocess_brain(patient_ids: range, patient_info: dict, spacing_target: list, folder: str, folder_out: str, get_patient_folder: Callable[[], str] , get_fname: Callable[[], str] , skip: list = [], verbose: bool = False) -> None: 
    """
    For each given patient, it calls process_image and places the output in a new structured tree with root folder_out.\n
    preprocess_brain() also updates the patient_info[id]["processed_shapes"] the reflect the new shape from the preprocess.

    Parameters:
    -----------
        patient_ids: range / list
            The ids of the patients to be preprocessed
        patient_info: dict
            The generated patient info from generate_patient_info()
        spacing_target: 
            The target spacing, usually given by spacing_target()
        folder: str
            The initial location of the patients
        folder_out: str
            The output location 
        get_patient_folder: Callable[[], str]
            A function returning the custom name of the patients folder
        get_fname: Callable[[], str]
            A function returning the custom name of the patients masks
        skip: list
            A list of the ids to skip , optional
        verbose: bool
            When True, will display progress every 10 patients preprocessed

    """
    #TODO : make sure this line below works fine
    patient_ids = [i for i in patient_ids if i not in skip]
    for id in patient_ids:
        patient_folder = get_patient_folder(folder, id)
        images = []
        fname = get_fname()
        fname = os.path.join(patient_folder, fname)
        if(not os.path.isfile(fname)):
            continue
        image, processed_shape = preprocess_image(
            nib.load(fname).get_fdata().astype(int),
            patient_info[id]["crop"],
            patient_info[id]["spacing"],
            spacing_target
        )
        # If depth of image changes after processing, it should be recorded
        patient_info[id]["processed_shape"] = processed_shape
        images.append(image)
        images = np.vstack(images)
        np.save(os.path.join(folder_out, "patient{:03d}".format(id)), images.astype(np.float32))
        if(verbose and id%10 == 0) : 
            print("Finished processing patient {:03d}".format(id))
    np.save(os.path.join(folder_out, "patient_info"), patient_info)
    

###########
##Dataset##
###########

class AddPadding(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def resize_image_by_padding(self, image, new_shape, pad_value=0):
        shape = tuple(list(image.shape))
        new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
        if pad_value is None:
            if len(shape) == 2:
                pad_value = image[0, 0]
            elif len(shape) == 3:
                pad_value = image[0, 0, 0]
            else:
                raise ValueError("Image must be either 2 or 3 dimensional")
        res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
        start = np.array(new_shape) / 2. - np.array(shape) / 2.
        if len(shape) == 2:
            res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
        elif len(shape) == 3:
            res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
            int(start[2]):int(start[2]) + int(shape[2])] = image
        return res
  
    def __call__(self, sample):
        sample = self.resize_image_by_padding(sample, new_shape=self.output_size)
        return sample

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def center_crop_2D_image(self, img, crop_size):
        if all(np.array(img.shape) <= crop_size):
            return img
        center = np.array(img.shape) / 2.
        if type(crop_size) not in (tuple, list):
            center_crop = [int(crop_size)] * len(img.shape)
        else:
            center_crop = crop_size
            assert len(center_crop) == len(img.shape)
        return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]

    def __call__(self, sample):
        sample = self.center_crop_2D_image(sample, crop_size=self.output_size)
        return sample

class OneHot(object):
    def one_hot(self, seg, num_classes=4):
        return np.eye(num_classes)[seg.astype(int)].transpose(2,0,1)
    def __call__(self, sample):
        sample = self.one_hot(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        sample = torch.from_numpy(sample).float()
        return sample

class MirrorTransform():
    def __call__(self, sample):
        if np.random.uniform() < 0.5:
            sample = np.copy(sample[::-1])
        if np.random.uniform() < 0.5:
            sample = np.copy(sample[:, ::-1])
        return sample

class SpatialTransform():
    def __init__(self, patch_size, do_elastic_deform=False, alpha=None, sigma=None,
        do_rotation=True, angle_x=(-np.pi/6,np.pi/6), angle_y=None, angle_z=None,
        do_scale=True, scale=(0.7, 1.4), border_mode_data='constant', border_cval_data=0, order_data=3,
        border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
        p_scale_per_sample=1, p_rot_per_sample=1,independent_scale_for_each_axis=False, p_rot_per_axis:float=1,
        p_independent_scale_per_axis: int=1):
        
        self.params = locals()
        self.params.pop("self")
        self.params["patch_center_dist_from_border"] = list(np.array(patch_size) // 2)

    def __call__(self, sample):
        sample = sample[None,None,:,:]
        _,sample = augment_spatial(sample, sample, **self.params) 
        return sample[0,0]
    
class ACDCPatient(torch.utils.data.Dataset):
    """
    The Dataset class representing one patient.

    Attributes
    ----------
        id: int
            The id of the patient. It is a unique identifier
        info: dict
            The corresponding item from patient_info. See generate_patient_info for details
        transform: torchvision.transforms.transforms.Compose
            The transforms applied to the patient data

    Methods
    -------
        __len__
            Returns the number of layers if the data matrix
        __getitem__
            Takes the specified layer (by the slice_id parameter), applies transformation and returns it
        
    """
    def __init__(self, root_dir, patient_id, transform=None):
        self.root_dir = root_dir
        self.id = patient_id
        self.info = np.load("preprocessed/patient_info.npy", allow_pickle=True).item()[patient_id]
        self.transform = transform

    def __len__(self):
        return self.info["shape_ED"][2] + self.info["shape_ES"][2]

#TODO : remove prints
    def __getitem__(self, slice_id):
        data = np.load(os.path.join(self.root_dir, "patient{:03d}.npy".format(self.id)))
        print("data shape : {}".format(np.shape(data)))
        print("slice_id = {}".format(slice_id))
        sample = data[slice_id]
        if self.transform:
            sample = self.transform(sample)
        return sample

class ACDCDataLoader():
    def __init__(self, root_dir, patient_ids, batch_size=None, transform=None):
        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.batch_size = batch_size
        self.transform = transform
        self.patient_loaders = []
        if batch_size is not None:
            for id in self.patient_ids:
                self.patient_loaders.append(torch.utils.data.DataLoader(
                    ACDCPatient(root_dir, id, transform=transform),
                    batch_size=batch_size, shuffle=False, num_workers=0
                ))
        self.counter_id = 0
    
    def set_batch_size(self, batch_size):
        self.patient_loaders = []
        for id in self.patient_ids:
            self.patient_loaders.append(torch.utils.data.DataLoader(
                ACDCPatient(self.root_dir, id, transform=self.transform),
                batch_size=batch_size, shuffle=False, num_workers=0
            ))
    
    def set_transform(self, transform):
        self.transform = transform
        for loader in self.patient_loaders:
            loader.dataset.transform = transform

    def __iter__(self):
        self.counter_iter = 0
        return self

    def __next__(self):
        if(self.counter_iter == len(self)):
            raise StopIteration
        loader = self.patient_loaders[self.counter_id]
        self.counter_id += 1
        self.counter_iter += 1
        if self.counter_id%len(self) == 0:
            self.counter_id = 0
        return loader

    def __len__(self):
        return len(self.patient_ids)

    def current_id(self):
        return self.patient_ids[self.counter_id]


############################
##Brain Dataset and Loader##
############################
class SYNDalaLoader() :
    """
    Loops through the patients, layer by layer.


    """
    def __init__(self, root_dir, patient_ids, batch_size=None, transform=None):
        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.batch_size = batch_size
        self.transform = transform
        self.patient_loaders = []
        if batch_size is not None:
            for id in self.patient_ids:
                self.patient_loaders.append(torch.utils.data.DataLoader(
                    SYNPatient(root_dir, id, transform=transform),
                    batch_size=batch_size, shuffle=False, num_workers=0
                ))
        self.counter_id = 0

    def set_batch_size(self, batch_size):
        self.patient_loaders = []
        for id in self.patient_ids:
            self.patient_loaders.append(torch.utils.data.DataLoader(
                SYNPatient(self.root_dir, id, transform=self.transform),
                batch_size=batch_size, shuffle=False, num_workers=0
            ))
    
    def set_transform(self, transform):
        self.transform = transform
        for loader in self.patient_loaders:
            loader.dataset.transform = transform

    def __iter__(self):
        self.counter_iter = 0
        return self

    def __next__(self):
        if(self.counter_iter == len(self)):
            raise StopIteration
        loader = self.patient_loaders[self.counter_id]
        self.counter_id += 1
        self.counter_iter += 1
        if self.counter_id%len(self) == 0:
            self.counter_id = 0
        return loader

    def __len__(self):
        return len(self.patient_ids)

    def current_id(self):
        return self.patient_ids[self.counter_id]

class SYNPatient(torch.utils.data.Dataset):
    """
    The Dataset class representing one patient.

    Attributes
    ----------
        id: int
            The id of the patient. It is a unique identifier
        info: dict
            The corresponding item from patient_info. See generate_patient_info for details
        transform: torchvision.transforms.transforms.Compose
            The transforms applied to the patient data

    Methods
    -------
        __len__
            Returns the number of layers if the data matrix
        __getitem__
            Takes the specified layer (by the slice_id parameter), applies transformation and returns it
        
    """

    def __init__(self, root_dir, patient_id, transform=None):
        self.root_dir = root_dir
        self.id = patient_id
        self.info = np.load(os.path.join(root_dir,"patient_info.npy"), allow_pickle=True).item()[patient_id]
        self.transform = transform

    def __len__(self):
        return self.info["processed_shape"][0]

    def __getitem__(self, slice_id):
        data = np.load(os.path.join(self.root_dir, "patient{:03d}.npy".format(self.id)))
        sample = data[slice_id]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
###########
##Testing##
###########

def evaluate_metrics(prediction, reference, keys: list = None):
    # NB : for the heart, keys = ["_RV", "_MYO", "_LV"]
    results = {}
    if keys == None:
        ref = np.copy(reference)
        pred = np.copy(prediction)
        try:
            results["DSC"] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["DSC"] = 0
        try:
            results["HD"] = binary.hd(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
        except:
            results["HD"] = np.nan
    else : 
        for c,key in enumerate(keys,start=1):
            ref = np.copy(reference)
            pred = np.copy(prediction)

            ref = ref if c==0 else np.where(ref!=c, 0, ref)
            pred = pred if c==0 else np.where(np.rint(pred)!=c, 0, pred)

            try:
                results["DSC" + key] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
            except:
                results["DSC" + key] = 0
            try:
                results["HD" + key] = binary.hd(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
            except:
                results["HD" + key] = np.nan
    return results
        

# def evaluate_metrics(prediction, reference):
#     results = {}
#     for c,key in enumerate(["_RV", "_MYO", "_LV"],start=1):
#         ref = np.copy(reference)
#         pred = np.copy(prediction)

#         ref = ref if c==0 else np.where(ref!=c, 0, ref)
#         pred = pred if c==0 else np.where(np.rint(pred)!=c, 0, pred)

#         try:
#             results["DSC" + key] = binary.dc(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
#         except:
#             results["DSC" + key] = 0
#         try:
#             results["HD" + key] = binary.hd(np.where(ref!=0, 1, 0), np.where(np.rint(pred)!=0, 1, 0))
#         except:
#             results["HD" + key] = np.nan
#     return results

def postprocess_image(image: np.array, info: dict, phase:str, current_spacing:list):
    """
    Takes the input image along with some information, and applies reverse transformation from the preprocessing step.

    Parameters
    ----------
        image: np.array
            The generated output from the auto encoder
        info: dict
            The corresponding patient_info to that image
        phase: str
            The image phase ("ED" or "ES")
        current_spacing: list
            The current image spacing, which corresponds to the median_spacing_target() of the training.
    """

    postprocessed = np.zeros(info["shape_{}".format(phase)])
    crop = info["crop_{}".format(phase)]
    original_shape = postprocessed[crop].shape
    original_spacing = info["spacing"]
    tmp_shape = tuple(np.round(original_spacing[1:] / current_spacing[1:] * original_shape[:2]).astype(int)[::-1])
    image = np.argmax(image, axis=1)
    image = np.array([torchvision.transforms.Compose([
            AddPadding(tmp_shape), CenterCrop(tmp_shape), OneHot()
        ])(slice) for slice in image]
    )
    image = resize_segmentation(image.transpose(1,3,2,0), image.shape[1:2]+original_shape,order=1)
    image = np.argmax(image, axis=0)
    postprocessed[crop] = image
    return postprocessed

#TODO : Generalise postprocessing
def postprocess_image_brain(image, info, current_spacing):
    """
    Takes the input image along with some information, and applies reverse transformation from the preprocessing step.

    Parameters
    ----------
        image: np.array
            The generated output from the auto encoder
        info: dict
            The corresponding patient_info to that image
        current_spacing: list
            The current image spacing, which corresponds to the median_spacing_target() of the training.
    """

    shape = info["shape"]
    shape = shape if len(info["shape"]) == 3 else shape[:3]

    postprocessed = np.zeros(shape)
    crop = info["crop"][:3]
    original_shape = postprocessed[crop].shape
    original_spacing = info["spacing"]
    tmp_shape = tuple(np.round(original_spacing[1:] / current_spacing[1:] * original_shape[:2]).astype(int)[::-1])
    image = np.argmax(image, axis=1)
    image = np.array([torchvision.transforms.Compose([
            AddPadding(tmp_shape), CenterCrop(tmp_shape), OneHot()
        ])(slice) for slice in image]
    )
    image = resize_segmentation(image.transpose(1,3,2,0), image.shape[1:2]+original_shape,order=1)
    image = np.argmax(image, axis=0)
    postprocessed[crop] = image
    return postprocessed
  
def testing(ae, test_loader, patient_info, folder_predictions, folder_out, current_spacing):
    ae.eval()
    with torch.no_grad():
        results = {"ED": {}, "ES": {}}
        for patient in test_loader:
            id = patient.dataset.id
            prediction, reconstruction = [], []
            for batch in patient: 
                batch = {"prediction": batch.to(device)}
                batch["reconstruction"] = ae.forward(batch["prediction"])
                prediction = torch.cat([prediction, batch["prediction"]], dim=0) if len(prediction)>0 else batch["prediction"]
                reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction)>0 else batch["reconstruction"]
            prediction = {
                "ED": prediction[:len(prediction)//2].cpu().numpy(),
                "ES": prediction[len(prediction)//2:].cpu().numpy()
            }
            reconstruction = {
                "ED": reconstruction[:len(reconstruction)//2].cpu().numpy(),
                "ES": reconstruction[len(reconstruction)//2:].cpu().numpy()
            }

            for phase in ["ED","ES"]:
                reconstruction[phase] = postprocess_image(reconstruction[phase], patient_info[id], phase, current_spacing)
                results[phase]["patient{:03d}".format(id)] = evaluate_metrics(
                    nib.load(os.path.join(folder_predictions, "patient{:03d}_{}.nii.gz".format(id, phase))).get_fdata(),
                    reconstruction[phase], 
                    keys = ["_RV", "_MYO", "_LV"]
                )
                nib.save(
                    nib.Nifti1Image(reconstruction[phase], patient_info[id]["affine"], patient_info[id]["header"]),
                    os.path.join(folder_out, 'patient{:03d}_{}.nii.gz'.format(id, phase))
                )
    return results

#TODO: Generalize testing

def testing_brain(ae, test_loader, patient_info, folder_predictions, folder_out, current_spacing):
    ae.eval()
    with torch.no_grad():
        results = {}
        for patient in test_loader:
            id = patient.dataset.id
            prediction, reconstruction = [], []
            for batch in patient: 
                batch = {"prediction": batch.to(device)}
                batch["reconstruction"] = ae.forward(batch["prediction"])
                prediction = torch.cat([prediction, batch["prediction"]], dim=0) if len(prediction)>0 else batch["prediction"]
                reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction)>0 else batch["reconstruction"]
            prediction = prediction.cpu().numpy(),
            reconstruction = reconstruction.cpu().numpy()

            reconstruction = postprocess_image_brain(reconstruction, patient_info[id], current_spacing)
            folder_out_patient = os.path.join(folder_out, "patient{:03d}".format(id))

            results["patient{:03d}".format(id)] = evaluate_metrics(
                nib.load(os.path.join(folder_predictions, f"patient{id:03d}", "mask.nii.gz")).get_fdata(),
                reconstruction,
                keys = None
            )
            if not os.path.exists(folder_out_patient) : os.makedirs(folder_out_patient)
            
            nib.save(
                nib.Nifti1Image(reconstruction, patient_info[id]["affine"], patient_info[id]["header"]),
                os.path.join(folder_out_patient,'mask.nii.gz')
            )
    return results
  
def display_image(img):
    img = np.rint(img)
    img = np.rint(img / 3 * 255)
    display(Image.fromarray(img.astype(np.uint8)))
  
def display_difference(prediction, reference):
    difference = np.zeros(list(prediction.shape[:2]) + [3])
    difference[prediction != reference] = [240,52,52]
    display(Image.fromarray(difference.astype(np.uint8)))
  
class Count_nan():
    def __init__(self):
        self.actual_nan = 0
        self.spotted_CA = 0
        self.FP_CA = 0
        self.total = 0
      
    def __call__(self, df): 
        df_AE = df[[column for column in df.columns if "p" in column]]
        df_GT = df[[column for column in df.columns if "p" not in column]]
        check_AE = np.any(np.isnan(df_AE.values), axis=1)
        check_GT = np.any(np.isnan(df_GT.values), axis=1)

        self.actual_nan += np.sum(check_GT)
        self.spotted_CA += np.sum(np.logical_and(check_GT, check_AE))
        self.FP_CA += np.sum(np.logical_and(np.logical_not(check_GT), check_AE))
        self.total += np.sum(np.any(np.isnan(df.values), axis=1))
      
    def __str__(self):
        string = "Anomalies (DSC=0/HD=nan): {}\n".format(self.actual_nan)
        string += "Spotted by CA: {}\n".format(self.spotted_CA)
        string += "False Positive by CA: {}\n".format(self.FP_CA)
        string += "Total discarded from the next plots: {}".format(self.total)
        return string
   
def process_results(models, folder_GT, folder_pGT):
    count_nan = Count_nan()
    plots = {}
    for model in models:
        GT = np.load(os.path.join(folder_GT, "{}.npy".format(model)), allow_pickle=True).item()
        pGT = np.load(os.path.join(folder_pGT, "{}_AE.npy".format(model)), allow_pickle=True).item()
        for phase in ["ED","ES"]:
            df = pd.DataFrame.from_dict(GT[phase], orient='index', columns=["DSC_LV", "HD_LV", "DSC_RV", "HD_RV", "DSC_MYO", "HD_MYO"])
            for measure in list(df.columns):
                df["p{}".format(measure)] = df.index.map({
                    patient: pGT[phase][patient][measure] for patient in pGT[phase].keys()
                })

            df = df.replace(0, np.nan)
            count_nan(df)
            df = df.dropna()

            for measure in ["DSC", "HD"]:
                for label in ["LV", "RV", "MYO"]:
                    if("GT_{}_{}".format(measure,label) not in plots.keys()):
                        plots["GT_{}_{}".format(measure,label)] = []
                        plots["pGT_{}_{}".format(measure,label)] = []
                    plots["GT_{}_{}".format(measure,label)] += list(df["{}_{}".format(measure,label)])
                    plots["pGT_{}_{}".format(measure,label)] += list(df["p{}_{}".format(measure,label)])
    print(count_nan)
    return plots
  
def display_plots(plots):
    plt.rcParams['xtick.labelsize'] = 30#'x-large'
    plt.rcParams['ytick.labelsize'] = 30#'x-large'
    plt.rcParams['legend.fontsize'] = 30#'x-large'
    plt.rcParams['axes.labelsize'] = 30#'x-large'
    plt.rcParams['axes.titlesize'] = 35#'x-large'

    grid = np.zeros([700*2, 700*3, 4])

    for i,measure in enumerate(["DSC", "HD"]):
        for j,label in enumerate(["LV", "RV", "MYO"]):
            x = "GT_{}_{}".format(measure, label)
            y = "pGT_{}_{}".format(measure, label)
            limx = np.ceil(max(plots[x] + plots[x]) / 10)*10 if measure=="HD" else 1
            limy = np.ceil(max(plots[y] + plots[y]) / 10)*10 if measure=="HD" else 1

            correlation = stats.pearsonr(plots[x], plots[y])[0]

            fig,axis = plt.subplots(ncols=1, figsize=(7, 7), dpi=100)
            sns.scatterplot(data=plots, x=x, y=y, ax=axis, label="Ours: r={:.3f}".format(correlation), color="blue", s=50)
            plt.plot(np.linspace(0, limx), np.linspace(0, limx), '--', color="gray", linewidth=5)

            axis.set_xlabel(measure)
            axis.set_ylabel("p{}".format(measure))
            axis.set_xlim([0, max(limx, limy)])
            axis.set_ylim([0, max(limx, limy)])
            axis.set_title(label)

            plt.grid()
            plt.tight_layout()
            plt.savefig("tmp.png")
            plt.close(fig)

            grid[i*700:(i+1)*700, j*700:(j+1)*700, :] = np.asarray(Image.open("tmp.png"))

    os.remove("tmp.png")
    grid = Image.fromarray(grid.astype(np.uint8))
    display(grid.resize((900,600), resample=Image.LANCZOS))
