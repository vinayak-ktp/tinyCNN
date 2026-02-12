import os
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path, target_size=(64, 64), normalize=True):
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    img_array = np.array(img)

    img_array = img_array.transpose(2, 0, 1)

    if normalize:
        img_array = img_array.astype(np.float32) / 255.0

    return img_array


def load_dataset_from_directory(directory, target_size=(64, 64), normalize=True, max_samples=None):
    # directory structure : directory/class1/*
    #                                /class2/*
    directory = Path(directory)
    class_names = sorted([d.name for d in directory.iterdir() if d.is_dir()])

    images = []
    labels = []

    for label, class_name in enumerate(class_names):
        class_dir = directory / class_name
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))

        if max_samples:
            image_files = image_files[:max_samples]

        for img_path in image_files:
            try:
                img = load_image(img_path, target_size, normalize)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels, class_names


def train_val_split(images, labels, val_split=0.2, shuffle=True, seed=42):
    n_samples = len(images)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    split_idx = int(n_samples * (1 - val_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    X_train = images[train_indices]
    y_train = labels[train_indices]
    X_val = images[val_indices]
    y_val = labels[val_indices]

    return X_train, y_train, X_val, y_val


def create_batches(X, y, batch_size=32, shuffle=True):
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        yield X[batch_indices], y[batch_indices]


def augment_image(image, flip_horizontal=True, rotation_range=10):
    img_array = image.transpose(1, 2, 0)
    img = Image.fromarray((img_array * 255).astype(np.uint8))

    if flip_horizontal and np.random.rand() > 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        img = img.rotate(angle)

    img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0

    return img_array


def get_dataset_stats(images, labels, class_names):
    stats = {
        'n_samples': len(images),
        'n_classes': len(class_names),
        'class_names': class_names,
        'image_shape': images[0].shape,
        'samples_per_class': {}
    }

    for i, class_name in enumerate(class_names):
        count = np.sum(labels == i)
        stats['samples_per_class'][class_name] = count

    return stats


def generate_synthetic_data(n_samples=1000, image_size=(32, 32), n_classes=2):
    X = []
    y = []

    for i in range(n_samples):
        img = np.zeros((3, image_size[0], image_size[1]))

        class_id = i % n_classes

        if class_id == 0:
            # Bright top half
            img[:, :image_size[0]//2, :] = np.random.uniform(0.5, 1.0, (3, image_size[0]//2, image_size[1]))
            img[:, image_size[0]//2:, :] = np.random.uniform(0.0, 0.3, (3, image_size[0]//2, image_size[1]))
        else:
            # Bright bottom half
            img[:, :image_size[0]//2, :] = np.random.uniform(0.0, 0.3, (3, image_size[0]//2, image_size[1]))
            img[:, image_size[0]//2:, :] = np.random.uniform(0.5, 1.0, (3, image_size[0]//2, image_size[1]))

        img += np.random.normal(0, 0.05, img.shape)
        img = np.clip(img, 0, 1)

        X.append(img)
        y.append(class_id)

    return np.array(X), np.array(y)
