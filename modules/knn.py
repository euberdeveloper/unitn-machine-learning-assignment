import torch
import torchvision
from torchvision import datasets, transforms

from pathlib import Path
from sklearn import neighbors

from .utils import merge_transformations, evalutate_helper
from .custom_transformations import SrotolaTransform, PercColoriTransform

class Knn:
    # Define the two phases (train, validate)
    phases = ['train', 'validation']
    # Define the datasets folder
    data_dir = Path('dataset')

    __init__(self, grayscale: bool, resize_scale: int, augmentation: bool, use_perc_col: bool, n_trees: int, criterion: str):
        self.n_trees = n_trees
        self.criterion = criterion

        # Get cuda device if available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Define the data transforms for the two phases. Note that they are based on the params given to the constructor
        data_transforms = {
            'train': merge_transformations([
                transforms.Resize(resize_scale),
                transforms.Grayscale() if grayscale else None,
                transforms.RandomHorizontalFlip() if augmentation else None,
                transforms.RandomVerticalFlip() if augmentation else None,
                transforms.ToTensor(),
                PercColoriTransform() if use_perc_col else SrotolaTransform()
            ]),
            'validation': merge_transformations([
                transforms.Resize(resize_scale),
                transforms.Grayscale() if grayscale else None,
                transforms.ToTensor(),
                PercColoriTransform() if use_perc_col else SrotolaTransform()
            ])
        }

    def create_datasets(self):
        print('Creating image datasets')
        # Create the image datasets
        self.image_datasets = {
            phase: datasets.ImageFolder(
                self.data_dir.join(phase), self.data_transforms[phase])
            for phase in self.phases
        }
        # Define the length of each class
        self.datasets_sizes = {
            phase: len(self.image_datasets[phase])
            for phase in self.phases
        }
        # Define the class names
        self.class_names = image_datasets['train'].classes
        print('Created image datasets')

    def create_dataloaders(self):
        print('Creating dataloaders')
        # Create the dataloaders. Note that the batch in unique per phase
        self.dataloaders = {
            phase: torch.utils.data.DataLoader(
                self.image_datasets[phase], batch_size=self.datasets_sizes[phase], shuffle=True, num_workers=2)
            for phase in self.phases
        }
        print('Created dataloaders')

    def get_total_images(self):
        print('Getting all the images')
        parsed_images = {
            phase: (images, labels)
            for phase in self.phases
            for images, labels in self.dataloaders[phase]
        }
        print('Getted all the images')

    def train(self):
        print('Training RForest')
        (train_images, train_labels) = self.parsed_images['train']
        train_images = train_images
        train_labels = train_labels
        self.model = neighbors.KNeighborsClassifier(n_jobs=-1)
        self.model = self.model.fit(train_images, train_labels)
        print('Trained RForest')

    def evaluate(self):
        validation_images, validation_labels = self.parsed_images['validation']
        total = len(validation_images)
        predicteds = self.model.predict(validation_images)
        data = zip(predicteds, validation_labels)
        evalutate_helper(self.class_names, total, data)
