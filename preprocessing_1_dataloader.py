import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets.utils as utils


def download_data():
    url_images = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    url_annotations = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    data_path = './data'

    utils.download_url(url_images, data_path)
    utils.extract_archive(os.path.join(data_path, "images.tar.gz"), data_path)

    utils.download_url(url_annotations, data_path)
    utils.extract_archive(os.path.join(data_path, "annotations.tar.gz"), data_path)


#Each pixel in a mask image can take one of three values: 1, 2, or 3. 1 means that this pixel of an image belongs to the class pet, 2 - to the class background, 3 - to the class border. 
def preprocess_mask(mask):
    
    mask[np.round(mask*255) == 2.0] = 0.0
    mask[(np.round(mask*255) == 1.0) | (np.round(mask*255) == 3.0)] = 1.0
    return mask

class OxfordPetDataset_with_labels(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform_data=None, transform_mask=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform_data = transform_data
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = Image.open(os.path.join(self.images_directory, image_filename))
        mask = Image.open(
            os.path.join(self.masks_directory, image_filename.replace(".jpg", ".png")),
        )
        
        if self.transform_data is not None:
            image = self.transform_data(image)
        
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)
        
        mask = preprocess_mask(mask)
        
        return image, mask

class OxfordPetDataset_with_labels_mixed(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, unlabeled_ratio=0.8, transform_data=None, transform_mask=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform_data = transform_data
        self.transform_mask = transform_mask
        self.unlabeled_ratio = unlabeled_ratio

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = Image.open(os.path.join(self.images_directory, image_filename))
        mask = Image.open(
            os.path.join(self.masks_directory, image_filename.replace(".jpg", ".png")),
        )
        
        if self.transform_data is not None:
            image = self.transform_data(image)
        
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)
        
        mask = preprocess_mask(mask)
        
        # randomly change 80% of masks to -1
        if random.random() < self.unlabeled_ratio:
            mask = torch.ones((1, image.shape[1], image.shape[2])) * -1
        
        return image, mask


def readable_images(images_filenames, images_directory):
    """
    Remove the data that are not readable
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

def get_data(nb_labeled_data, nb_unlabeled_data, percentage_validation, percentage_test):
    """
    nb_labeled_data : number of labeled data
    nb_unlabeled_data : number of unlabeled data
    percentage_validation : percentage from the whole dataset of the validation set
    percentage_test : percentage from the whole dataset of the test set
    Output:
    dataloader for labeled training data
    dataloader for unlabeled training data
    dataloader for validation set
    dataloader for test set
    """

    assert nb_labeled_data + nb_unlabeled_data == 1

    # download_data() # Comment out if you have already this downloaded

    images_directory = os.path.join("./data/images")
    masks_directory = os.path.join("./data/annotations/trimaps")

    images_filenames = list(sorted(os.listdir(images_directory)))
    print('all images are = ', len(images_filenames))
    
    correct_images_filenames = readable_images(images_filenames, images_directory)
    correct_images_filenames = are_images_all_RGB(correct_images_filenames, images_directory)

    random.shuffle(correct_images_filenames)

    
    nb_data = len(correct_images_filenames)
    
    transform_data = transforms.Compose(
        [transforms.ToTensor(),
        #transforms.CenterCrop((256, 256))])
        transforms.Resize((256,256))])
        #transforms.Normalize((0, 0, 0), (1/255, 1/255, 1/255))])
    
    
    transform_mask = transforms.Compose(
        [transforms.ToTensor(),
        #transforms.CenterCrop((256, 256))])
        transforms.Resize((256,256))])
        #transforms.Normalize(0, 1/255)])

    ##train data
    index_end_train = int((1 - (percentage_validation + percentage_test)) * nb_data)
    train_images_filenames = correct_images_filenames[0:index_end_train]

    random.shuffle(train_images_filenames)

    nb_data_train = len(train_images_filenames)
    labeled_train_images_filenames = train_images_filenames[0:int(nb_labeled_data * nb_data_train)]
    unlabeled_train_images_filenames = train_images_filenames[int(nb_labeled_data * nb_data_train):]

    #train mixed labeled and unlabeled data
    mixed_data_train = OxfordPetDataset_with_labels_mixed(train_images_filenames, images_directory, masks_directory, nb_unlabeled_data, transform_data, transform_mask)
    mixed_train_loader = DataLoader(
        mixed_data_train,
        batch_size=100,
        shuffle=True,
    )

    
    ##validation data
    index_start_val = index_end_train
    index_end_val = index_end_train + int(percentage_validation * nb_data)
    validation_images_filenames = correct_images_filenames[index_start_val:index_end_val]

    validation_data = OxfordPetDataset_with_labels(validation_images_filenames, images_directory, masks_directory,transform_data,transform_mask)
    val_loader = DataLoader(
        validation_data,
        batch_size=20,
        shuffle=True,
    )

    ##test data 
    index_start_test = index_end_val
    test_images_filenames = correct_images_filenames[index_start_test :]

    test_data = OxfordPetDataset_with_labels(test_images_filenames, images_directory, masks_directory,transform_data,transform_mask)
    test_loader = DataLoader(
        test_data,
        batch_size=20,
        shuffle=True,
    )

    return mixed_train_loader, val_loader, test_loader