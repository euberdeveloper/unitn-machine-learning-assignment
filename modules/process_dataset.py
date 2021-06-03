from pathlib import Path
import random
import shutil

def process_dataset(validation_percentage = 0.20) -> None:
    # Get train and validation dirs
    dataset_dir = Path('dataset')
    train_dir = dataset_dir / 'train'
    validation_dir = dataset_dir / 'validation'

    # Get the classes' dirs from the train dir
    class_dirs = [
        class_dir
        for class_dir in train_dir.iterdir()
        if class_dir.is_dir()
    ]

    # For every class dir
    for class_dir in class_dirs:
        # Get the name of the class and print it
        class_name = class_dir.name
        print(f'Class {class_name}')

        # Get the files of that class
        class_files = [
            class_file
            for class_file in class_dir.iterdir()
            if class_file.is_file()
        ]

        # Count the number of samples for that class
        class_files_size = len(class_files)

        # Randomly get "validation_percentage" samples that will be used for validation purposes
        n_validation_files = round(class_files_size * (validation_percentage))
        validation_files = random.sample(class_files, n_validation_files)

        # For each validation file, move it to the validation dir
        for validation_file in validation_files:
            from_path = validation_file
            to_path = validation_dir / validation_file.parent.name / validation_file.name
            to_path.mkdir(parents=True, exist_ok=True)
            shutil.move(str(from_path), str(to_path))

        # Print number of samples for train / validation datasets
        print(f'Train dir containing {len(list((train_dir / class_name).iterdir()))}')
        print(f'Validation dir containing {len(list((validation_dir / class_name).iterdir()))}')