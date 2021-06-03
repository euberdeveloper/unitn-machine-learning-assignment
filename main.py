from .modules.process_dataset import process_dataset
from .modules.decision_tree import DTree
from .modules.random_forest import RandomForest
from .modules.knn import Knn
from .modules.svm import Svm
from .modules.nn import NN

# PROCESS DATASET (ONLY FIRST TIME)

def p_dataset():
    process_dataset()

# DT

def decision_tree_grayscale():
    model = DTree(grayscale=True, resize_scale=72, augmentation=True, use_perc_col=False)
    model.create_datasets()
    model.create_dataloaders()
    model.get_total_images()
    model.train()
    model.evaluate()

def decision_tree():
    model = DTree(grayscale=False, resize_scale=72, augmentation=True, use_perc_col=False)
    model.create_datasets()
    model.create_dataloaders()
    model.get_total_images()
    model.train()
    model.evaluate()

# RANDOM FOREST

def random_forest_grayscale():
    model = RandomForest(grayscale=True, resize_scale=72, augmentation=True, use_perc_col=False, n_trees=20, criterion='entropy')
    model.create_datasets()
    model.create_dataloaders()
    model.get_total_images()
    model.train()
    model.evaluate()

def random_forest():
    model = RandomForest(grayscale=False, resize_scale=72, augmentation=True, use_perc_col=False, n_trees=20, criterion='entropy')
    model.create_datasets()
    model.create_dataloaders()
    model.get_total_images()
    model.train()
    model.evaluate()

# KNN

def knn_grayscale():
    model = Knn(grayscale=True, resize_scale=72, augmentation=True, use_perc_col=False)
    model.create_datasets()
    model.create_dataloaders()
    model.get_total_images()
    model.train()
    model.evaluate()

def knn():
    model = Knn(grayscale=False, resize_scale=72, augmentation=True, use_perc_col=False)
    model.create_datasets()
    model.create_dataloaders()
    model.get_total_images()
    model.train()
    model.evaluate()

# SVM

def svm_grayscale():
    model = Svm(grayscale=True, resize_scale=72, augmentation=True, use_perc_col=False)
    model.create_datasets()
    model.create_dataloaders()
    model.get_total_images()
    model.train()
    model.evaluate()

def svm():
    model = Svm(grayscale=False, resize_scale=72, augmentation=True, use_perc_col=False)
    model.create_datasets()
    model.create_dataloaders()
    model.get_total_images()
    model.train()
    model.evaluate()

# NN

def nn():
    model = Svm(num_epochs=23)
    model.create_datasets()
    model.create_dataloaders()
    model.get_sampler()
    model.train()