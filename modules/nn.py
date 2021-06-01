import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

import torchvision.transforms.functional as TF
import random
from typing import Sequence


class NN:
    # Define the two phases (train, validate)
    phases = ['train', 'validation']
    # Define the datasets folder
    data_dir = Path('dataset')

    __init__(self, num_epochs: int):
        self.num_epochs = num_epochs

        # Get cuda device if available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Define the normalizations
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Define the data transforms for the two phases. Note that they are based on the params given to the constructor
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                #CustomRotateTransform((90, 180, 270)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'validation': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
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

    def get_sampler(self):
        def get_sample_weights(dataset):
            class_counts = [1144, 187, 681, 1048,
                            796, 1135, 1037, 1480, 1472, 1024]
            num_samples = sum(class_counts)
            labels = []
            for _, label in dataset:
                labels.append(label)
            class_weights = [num_samples/class_counts[i]
                             for i in range(len(class_counts))]
            weights = [class_weights[labels[i]]
                       for i in range(int(num_samples))]
            return weights
        sample_weights = get_sample_weights(self.image_datasets['train'])
        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(
            sample_weights, len(sample_weights))

    def create_dataloaders(self):
        print('Creating dataloaders')
        # Create the dataloaders. Note that the batch in unique per phase
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=128, sampler=self.sampler, num_workers=2),
            'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=128, shuffle=True, num_workers=2),
        }
        print('Created dataloaders')

    def _train_model(model, criterion, optimizer, scheduler,):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs -1 }')
            print('-' * 10)

            since_epoch = time.time()

            for phase in self.phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_corrects_for_class = [0] * 10
            dataset_sizes_for_class = [0] * 10

            for inputs, labels in self.dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                for i in range(10):
                running_corrects_for_class[i] += torch.sum(
                    torch.logical_and(preds == labels.data, labels.data == i))
                dataset_sizes_for_class[i] += torch.sum(labels.data == i)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            acc_for_class = [el.double() / dataset_sizes_for_class[i]
                             for i, el in enumerate(running_corrects_for_class)]
            arr_names_for_class = [
                f'{name}: {val}' for name, val in zip(class_names, acc_for_class)]
            epoch_classes_acc = sum([el.double() / dataset_sizes_for_class[i]
                                    for i, el in enumerate(running_corrects_for_class)]) / 10
            print(f'{phase} Acc for class: {", ".join(arr_names_for_class)}')
            print(
                f'{phase} Loss: {epoch_loss:.4f} Accurancy: {epoch_acc:.4f} Class acc: {epoch_classes_acc:.4f}')

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            time_elapsed_epoch = time.time() - since_epoch
            print(
                f'Epoch took {time_elapsed_epoch//60:.0f}m {time_elapsed_epoch%60:.0f}s')
            print()

        time_elapsed = time.time() - since
        print(
            f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)
        return model

    def train(self):
        model = models.resnet50(pretrained=True)
        model.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.2, training=m.training))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes_length)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        learning_rate = 0.023
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        model  = train_model(model, criterion, optimizer, scheduler, num_epochs=10)

