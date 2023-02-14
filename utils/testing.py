import numpy as np
from medpy.metric import binary
import os
from typing import Callable
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import torchvision
import pandas as pd
import seaborn as sns
from IPython.display import display
from batchgenerators.augmentations.utils import resize_segmentation
from PIL import Image
from scipy import stats

from utils.dataset import AddPadding, CenterCrop, OneHot, DataLoader
from utils.preprocess import preprocess, median_spacing_target



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

def generate_testing_set(ae , data_path:str, alter_image:Callable, transform, test_ids:list='default' ):
    """
        Given a model 'alter_image', will generate model_GT, and run the AE on it to output a model_pGT"
    """


    # Creating the required paths
    required_paths = ["measures", "measures/preprocessed_model"]
    for path in required_paths:
        full_path = os.path.join(data_path, path)
        if not os.path.exists(full_path): os.makedirs(full_path)
    
    # Gathering info about the patients and dataset
    patient_info = np.load(os.path.join(data_path,'preprocessed/patient_info.npy'), allow_pickle=True).item()
    spacing = median_spacing_target(os.path.join(data_path, "preprocessed"), 2)
    optimal_parameters = np.load(os.path.join(data_path, "preprocessed", "optimal_parameters.npy"), allow_pickle=True).item()
    BATCH_SIZE = optimal_parameters["BATCH_SIZE"]
    test_ids = np.load(os.path.join(data_path, 'saved_ids.npy'), allow_pickle=True).item().get('test_ids') if test_ids == 'default' else test_ids

    # Creates nii.gz for the model_GT, and preprocesses that model_GT for the AE to run
    preprocess(
        data_path=data_path,
        verbose=False,
        alter_image=alter_image
        )
    

    test_loader = DataLoader(data_path, mode='test', batch_size=BATCH_SIZE, transform=transform)
    
    # Evaluates the model data with the trained AE
    _ = testing(
        ae=ae,
        test_loader=test_loader,
        patient_info=patient_info,
        folder_predictions=os.path.join(data_path, "measures/preprocessed_model"),
        folder_out=os.path.join(data_path, "measures/pGT"),
        current_spacing=spacing,
        compute_results=False)

def compute_results(data_path, test_ids = 'default', measures = 'hd'):
    # TODO : assert that required functions have been executed prior
    # TODO : use evaluate_metrics within compute_results
    """
    Computes plottable and relevant results to evaluate the AE's performance on model data.
    Returns (GT_to_model_GT, GT_to_model_pGT) as X,Y

    Parameters
    ----------
        data_path: str
            A string of the absolute or relative path to the specific root data folder eg. data/{application}
        test_ids: list or 'default'
            The list of the Ids to process, usually the testing set. Option 'default' will use the saved test_ids
    Additional details
    ------------------
    GT_to_model_GT contains metric evaluations for the real GT and the model generated GT
    GT_to_model_pGT contains metric evaluations between the real GT and the pGT generated from the model

    The idea is that, if a model performs well, GT and model_GT will be close --> Low distance between GT and model_pGT.
    If on the contratry a model performs badly, GT and model_GT will be different --> High distance between GT and model_pGT
    """
    assert measures in ['hd', 'dc', 'both'],"Parameter measures must be 'hd', 'dc' or 'both'"
    

    test_ids = np.load(os.path.join(data_path, 'saved_ids.npy'), allow_pickle=True).item().get('test_ids') if test_ids == 'default' else test_ids
    
    # Initialising results dictionary

    results = {}
    key_GT_to_model_GT, key_GT_to_model_pGT = 'GT_to_model_GT', 'GT_to_model_pGT'
    results[key_GT_to_model_GT], results[key_GT_to_model_pGT] = {}, {}

    # Finding out the measures used
    dc, hd = False, False
    
    if measures == 'hd' or measures == 'both' : 
        hd = True
        for key in list(results.keys()) :
            results[key]['hd'] = {}
    if measures == 'dc' or measures == 'both' : 
        dc = True
        for key in list(results.keys()):
            results[key]['dc'] = {}


    for id in test_ids:
        # Retrieving the paths of all the images to compute results from
        path_model_GT = os.path.join(data_path, 'measures/structured_model/patient{:03d}/mask.nii.gz').format(id)
        path_GT = os.path.join(data_path, 'structured/patient{:03d}/mask.nii.gz').format(id)
        path_model_pGT = os.path.join(data_path, 'measures/pGT/patient{:03d}/mask.nii.gz').format(id)
        # Retrieving the images
        model_GT = nib.load(path_model_GT).get_fdata()
        GT = nib.load(path_GT).get_fdata()
        model_pGT = nib.load(path_model_pGT).get_fdata()
        # Removing the time dimension
        if len(GT.shape) == 4: GT = GT[:, :, :, 0]
        if len(model_GT.shape) == 4: model_GT = model_GT[:, :, :, 0]
        if len(model_pGT.shape) == 4: model_pGT = model_pGT[:, :, :, 0]
        # Appending the results
        if dc :
            GT_to_model_GT = (binary.dc(np.where(GT!=0, 1, 0), np.where(np.rint(model_GT)!=0, 1, 0)))
            GT_to_model_pGT = (binary.dc(np.where(GT!=0, 1, 0), np.where(np.rint(model_pGT)!=0, 1, 0)))
            results[key_GT_to_model_GT]['dc'][id] = GT_to_model_GT
            results[key_GT_to_model_pGT]['dc'][id] = GT_to_model_pGT
        if hd : 
            GT_to_model_GT = (binary.hd(np.where(GT!=0, 1, 0), np.where(np.rint(model_GT)!=0, 1, 0)))
            GT_to_model_pGT = (binary.hd(np.where(GT!=0, 1, 0), np.where(np.rint(model_pGT)!=0, 1, 0)))
            results[key_GT_to_model_GT]['hd'][id] = GT_to_model_GT
            results[key_GT_to_model_pGT]['hd'][id] = GT_to_model_pGT
    np.save(os.path.join(data_path), 'results.npy', results)
    return results



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
def postprocess_image(image, info, current_spacing):
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
  

def testing(ae, data_path:os.PathLike, test_loader:DataLoader, patient_info:dict='default', folder_predictions:os.PathLike='default', folder_out:os.PathLike='default', current_spacing:list='default', compute_results:bool = True):
    # Option compute_results was added so that if needed, results don't have to be computed

    patient_info = np.load(os.path.join(data_path,'preprocessed/patient_info.npy'), allow_pickle=True).item() if patient_info=='default' else patient_info
    current_spacing = median_spacing_target(os.path.join(data_path), 'preprocessed') if current_spacing=='default' else current_spacing
    folder_predictions=os.path.join(data_path, 'structured') if folder_predictions==None else folder_predictions
    folder_out = os.path.join(data_path, 'reconstructions')if folder_out==None else folder_out
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

            reconstruction = postprocess_image(reconstruction, patient_info[id], current_spacing)
            folder_out_patient = os.path.join(folder_out, "patient{:03d}".format(id))
            if compute_results :
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

def process_results_single(results:dict):
    keys = list(results.keys())
    plots = {}
    for key in keys: # GT_to_model_GT or GT_to_model_pGT
        plots[key] = {}
        for measure_key in (list(results[key].keys())) : # hd or dc
            plots[key][measure_key] = list(results[key][measure_key].values())
                

    return plots

def plot_results(results):
    # TODO : Implement multiple keys support

    plots = process_results_single(results)
    measures = list(plots[list(plots.keys())[0]].keys())

    fig, axs = plt.subplots(len(measures),1)
    single = True if len(measures) == 1 else False
    for i,measure in enumerate(measures):
        x = plots['GT_to_model_GT'][measure]
        y = plots['GT_to_model_pGT'][measure]
        current_axe = axs[i] if not single else axs
        
        current_axe.scatter(x,y)
        current_axe.set_title(f'Results for {measure} :', loc='center')
        current_axe.grid()
        # if measure == 'dc':
        #     current_axe.set_xlim(left=0, right=1)
        #     current_axe.set_ylim(bottom=0, top=1)
    fig.suptitle("Plotted measures")
    fig.tight_layout()
   
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
