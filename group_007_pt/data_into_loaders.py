import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets.utils as utils

random.seed(200) # Fix randomness

# mean and std of whole image dataset
DATA_MEAN = torch.asarray([0.4803, 0.4497, 0.3960])
DATA_STD = torch.asarray([0.2665, 0.2623, 0.2707])

# Download images and annotations
def download_data():
    url_images = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    url_annotations = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    data_path = './data'

    utils.download_url(url_images, data_path)
    utils.extract_archive(os.path.join(data_path, "images.tar.gz"), data_path)

    utils.download_url(url_annotations, data_path)
    utils.extract_archive(os.path.join(data_path, "annotations.tar.gz"), data_path)


# Each pixel in a mask image can take one of three values: 1, 2, or 3. 
# 1: animal, 2: background, 3 border. 

# Change class 3 to 1 (i.e. account border as animal)
def preprocess_mask(mask):
    
    mask[mask == (2.0 / 255)] = 0.0
    mask[(mask == 1.0 / 255) | (mask == 3.0 / 255)] = 1.0
    return mask


class OxfordPetDataset_with_labels(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform_data_1=None, transform_mask_1=None, transform_2=None):
        """
        Constructor for the OxfordPetDataset_with_labels class.

        Parameters:
            images_filenames (list): List of image filenames
            images_directory (str): Path to the directory containing the images
            masks_directory (str): Path to the directory containing the masks
            transform_data_1 (callable, optional): Transform to apply to the image data
            transform_mask_1 (callable, optional): Transform to apply to the mask data
            transform_2 (callable, optional): Additional transform to apply to both the image and mask data
        """
        
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory

        self.transform_data_1 = transform_data_1
        self.transform_mask_1 = transform_mask_1
        self.transform_2 = transform_2

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.images_filenames)

    def __getitem__(self, idx):

        """
        Gets the image and mask data for the given index.

        Parameters:
            idx (int): The index of the data to retrieve.

        Returns:
            image (PIL.Image.Image): The image data.
            mask (PIL.Image.Image): The mask data.
        """

        # Load the image and mask
        image_filename = self.images_filenames[idx]
        image = Image.open(os.path.join(self.images_directory, image_filename))
        mask = Image.open(
            os.path.join(self.masks_directory, image_filename.replace(".jpg", ".png")),
        )
        
        # Apply first transforms to images and masks
        if self.transform_data_1 is not None:
            image = self.transform_data_1(image)
        if self.transform_mask_1 is not None:
            mask = self.transform_mask_1(mask)
        
        # Apply preprocess_mask function to make class 3 equal to 1 (i.e. border to animal)
        mask = preprocess_mask(mask)

        # Apply the second transforms to images and masks
        if self.transform_2 is not None:
            mask = self.transform_2(mask)
            image = self.transform_2(image)
        
        # Threshold the mask values to be either 0 or 1
        mask[mask <= 0.5] = 0
        mask[mask>0.5] = 1 
        
        return image, mask

# Class returning 
class OxfordPetDataset_with_labels_mixed(Dataset):
    def __init__(self, images_filenames, is_labelled_indeces, images_directory, masks_directory, unlabeled_ratio=0.8, transform_data_1=None, transform_mask_1=None, transform_2=None):
        """
        Custom PyTorch dataset that loads and preprocesses images and masks from specified directories.
        
        Args:
        - images_filenames (list): a list of image filenames
        - is_labelled_indeces (list): a binary list that indicates whether each image is labelled or not
        - images_directory (str): the directory where the images are stored
        - masks_directory (str): the directory where the masks are stored
        - unlabeled_ratio (float): the fraction of images that are unlabeled
        - transform_data_1 (callable): a transform to apply to the image data
        - transform_mask_1 (callable): a transform to apply to the mask data
        - transform_2 (callable): a transform to apply to both the image and mask data
        """
        
        self.images_filenames = images_filenames
        self.is_labelled_indeces = is_labelled_indeces
        self.images_directory = images_directory
        self.masks_directory = masks_directory

        self.transform_data_1 = transform_data_1
        self.transform_mask_1 = transform_mask_1
        self.transform_2 = transform_2
        
        self.unlabeled_ratio = unlabeled_ratio

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.images_filenames)

    def __getitem__(self, idx):
        """
        Loads the image and mask corresponding to the given index, applies the specified transforms,
        preprocesses the mask by thresholding it, and returns a tuple containing the preprocessed
        image and mask.
        
        Args:
        - idx (int): the index of the image/mask pair to load
        
        Returns:
        - a tuple containing the preprocessed image and mask
        """
        
        image_filename = self.images_filenames[idx]
        is_labelled_one = self.is_labelled_indeces[idx]
        
        # Load the image and mask
        image = Image.open(os.path.join(self.images_directory, image_filename))
        mask = Image.open(
            os.path.join(self.masks_directory, image_filename.replace(".jpg", ".png")),
        )
        
        # Apply first transforms to images and masks
        if self.transform_data_1 is not None:
            image = self.transform_data_1(image)
        if self.transform_mask_1 is not None:
            mask = self.transform_mask_1(mask)
        
        # Apply preprocess_mask function to make class 3 equal to 1 (i.e. border to animal)
        mask = preprocess_mask(mask)

        # Apply the second transforms to images and masks
        if self.transform_2 is not None:
            image = self.transform_2(image)
            mask = self.transform_2(mask)

        # Threshold the mask values to be either 0 or 1
        mask[mask <= 0.5] = 0
        mask[mask > 0.5] = 1 
        
        # If the image is unlabeled, set the mask to a tensor of ones with the same shape as the image tensor
        # multiplied by -1        
        if is_labelled_one == 0:
            mask = torch.ones((1, image.shape[1], image.shape[2])) * -1
        
        return image, mask


def readable_images(images_filenames, images_directory):
    """
    Remove the data that are not readable (i.e. cannot be opened)
    """
    correct_images_filenames = []
    
    for i in images_filenames:
        try : 
            Image.open(os.path.join(images_directory, i))
            correct_images_filenames.append(i)
        except:
            continue

    return correct_images_filenames

def are_images_all_RGB(images_filenames, images_directory):
    """
    Remove the data that do not have shape of (3,_,_)
    """
    correct_images_filenames = []
    transform = transforms.Compose([transforms.ToTensor()])
    for i in images_filenames:
        img = Image.open(os.path.join(images_directory, i))
        img = transform(img)
        if img.shape[0] == 3:
            correct_images_filenames.append(i)
    return correct_images_filenames


def un_normalise(X):
    """
    Given a batch or single image un-normalise it to get it to it's original values
    """

    if len(X.shape) == 4:
        shape_string = 'bchw'
    # given a single image
    else:
        shape_string = 'chw'
    
    original_mean = DATA_MEAN
    original_std = DATA_STD
    ones_like = torch.ones_like(X)
    # get mean into a shape we can easily add it to the images
    original_mean_broadcastable = torch.einsum(shape_string+',c->'+shape_string, ones_like, original_mean)

    # in each channel multiple by original std then add the original mean
    Y = torch.einsum(shape_string+',c->'+shape_string, X, original_std)
    Y += original_mean_broadcastable
    
    return Y


def get_data(nb_labeled_data, nb_unlabeled_data, percentage_validation, percentage_test, batch_size=16, img_resize=128, is_mixed_loader=True, pct_data = 1):
    """
    nb_labeled_data : number of labeled data
    nb_unlabeled_data : number of unlabeled data
    percentage_validation : percentage from the whole dataset of the validation set
    percentage_test : percentage from the whole dataset of the test set
    batch_size : (int) size of all data loader batches
    img_resize : (int) all images are resized to this size
    is_mixed_loader : (bool) if True the train loader only consits of labelled data (only for development)
    Output:
    dataloader for labeled and unlabelled training data in one loader
    dataloader for validation set
    dataloader for test set
    """
    
    random.seed(200) # Fix randomness
    
    assert nb_labeled_data + nb_unlabeled_data == 1

    download_data() # Comment out if you have already this downloaded

    images_directory = os.path.join("./data/images")
    masks_directory = os.path.join("./data/annotations/trimaps")

    images_filenames = list(sorted(os.listdir(images_directory)))
    print('all images are = ', len(images_filenames))
    
    correct_images_filenames = readable_images(images_filenames, images_directory)
    correct_images_filenames = are_images_all_RGB(correct_images_filenames, images_directory)

    random.shuffle(correct_images_filenames)

    nb_data = len(correct_images_filenames)
    
    transform_data_1 = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD)
        ])
    
    transform_mask_1 = transforms.Compose(
        [transforms.ToTensor()])
    
    transform_2 = transforms.Compose(
        [transforms.Resize((img_resize,img_resize))]
    )

    ##train data
    index_end_train = int((1 - (percentage_validation + percentage_test)) * nb_data)
    train_images_filenames = correct_images_filenames[:index_end_train]
    train_images_filenames = train_images_filenames[:int(len(train_images_filenames) * pct_data)]
    
    ##validation data
    index_start_val = index_end_train
    index_end_val = index_end_train + int(percentage_validation * nb_data)
    validation_images_filenames = correct_images_filenames[index_start_val:index_end_val]

    ##test data 
    index_start_test = index_end_val
    test_images_filenames = correct_images_filenames[index_start_test :]

    #Trained image names shuffled
    random.shuffle(train_images_filenames)

    # Create another list that stores the info whether each train_image_filename is unlabelled or not
    is_labelled = [0] * int(len(train_images_filenames)* nb_unlabeled_data) + [1] * int(nb_labeled_data*len(train_images_filenames)) # Doesnt have to be random as the images names are random anyway
    
    # If the lengths are not equal just add a 1 or remove the last number
    if len(is_labelled) > len(train_images_filenames):
      is_labelled.pop()
    if len(is_labelled) < len(train_images_filenames):
      is_labelled.append(1)
    random.shuffle(is_labelled)

    #train mixed labeled and unlabeled data:
    if is_mixed_loader:
        mixed_data_train = OxfordPetDataset_with_labels_mixed(train_images_filenames, is_labelled, images_directory, masks_directory, nb_unlabeled_data, transform_data_1, transform_mask_1, transform_2)
        mixed_train_loader = DataLoader(
            mixed_data_train,
            batch_size=batch_size,
            shuffle=False,
        )
  
    # if we only want labelled data in train_loader:
    else:
        mixed_data_train = OxfordPetDataset_with_labels(train_images_filenames, images_directory, masks_directory,transform_data_1, transform_mask_1, transform_2)
        mixed_train_loader = DataLoader(
            mixed_data_train,
            batch_size=batch_size,
            shuffle=False,
        )

    validation_data = OxfordPetDataset_with_labels(validation_images_filenames, images_directory, masks_directory,transform_data_1, transform_mask_1, transform_2)
    val_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
    )

    test_data = OxfordPetDataset_with_labels(test_images_filenames, images_directory, masks_directory,transform_data_1,transform_mask_1, transform_2)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return mixed_train_loader, val_loader, test_loader

