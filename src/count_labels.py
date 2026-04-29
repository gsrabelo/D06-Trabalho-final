from pathlib import Path
import numpy as np

train_label_dir = Path('data/dataset/train/labels')
val_label_dir = Path('data/dataset/val/labels')
test_label_dir = Path('data/dataset/test/labels')

def count_labels(label_dir_path,labels):
    label_dir = Path(label_dir_path)
    print (f'Counting labels in directory: {label_dir}')
    print(f'Looking for labels: {labels}')
    occurences_count = [0] * len(labels)
    images_count = [0] * len(labels)
    found = [False] * len(labels)
    total_files = 0
    for file_path in label_dir.glob('*.txt'):
        if file_path.is_file():
            total_files += 1
            with open(file_path, 'r') as f:
                #print(f'Processing file: {file_path}')
                lines = f.readlines()
                found = [False] * len(labels)
                for line in lines:
                    i = 0
                    for label in labels:
                        if line[0] == label:
                            occurences_count[i] += 1
                            if found[i] == False:
                                images_count[i] += 1
                                found[i] = True
                        i += 1
    return np.column_stack((np.array(occurences_count), np.array(images_count))), total_files               

if __name__ == '__main__':
    labels = ['0', '1']
    count, total_files = count_labels(train_label_dir, labels)
    print('='*60)
    print(f'Total files processed in {train_label_dir}: {total_files}')
    for i, label in enumerate(labels):
        print(f'Number of {label} labels in train: {count[i][0]}')
        print(f'Number of {label} images in train: {count[i][1]}')
    print('='*60)

    count, total_files = count_labels(val_label_dir, labels)
    print('='*60)
    print(f'Total files processed in {val_label_dir}: {total_files}')
    for i, label in enumerate(labels):
        print(f'Number of {label} labels in val: {count[i][0]}')
        print(f'Number of {label} images in val: {count[i][1]}')
    print('='*60)

    count, total_files = count_labels(test_label_dir, labels)
    print('='*60)
    print(f'Total files processed in {test_label_dir}: {total_files}')
    for i, label in enumerate(labels):
        print(f'Number of {label} labels in test: {count[i][0]}')
        print(f'Number of {label} images in test: {count[i][1]}')
    print('='*60)   