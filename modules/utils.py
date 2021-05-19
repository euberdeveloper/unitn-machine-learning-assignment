from torchvision import transforms

def merge_transformations(transformations: list):
    return transforms.Compose([t for t in transformations if t is not None])

def evalutate_helper(class_names: list[string], total: int, data):
    num_classes = len(class_names)

    total_for_class = [0] * num_classes
    correct = 0
    correct_for_class = [0] * num_classes

    for (predicted, label) in data:
        total_for_class[label] += 1
        if predicted == label:
            correct += 1
            correct_for_class[label] += 1

    accs_for_class = [
        c / t
        for t, c in zip(total_for_class, correct_for_class)
    ]
    acc_for_class = sum(accs_for_class) / num_classes
    
    print(f'Total samples: {total}')
    print(f'Correct samples: {correct}')
    print(f'Accurancy over samples: {correct / total}')

    print()

    print(f'Total for class: {zip(total_for_class, class_names)}')
    print(f'Correct for class: {zip(correct_for_class, class_names)}')
    print(f'Accs for class: {zip(accs_for_class, class_names)}')
    print(f'Accurancy over classes: {acc_for_class}')
