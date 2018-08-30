from typing import Tuple, List
from utils.global_config import (
    BATCH_SIZE,
    IMG_HEIGHT,
    IMG_WIDTH,
    PATH_TO_DATASET,
    LABEL,
    IMG_EXTENSIONS
)
import numpy as np
import cv2
import os


def process_raw_line_siamese(line: str) -> Tuple[str, str]:
    """Maps from line -> image paths

    Args:
        line: A raw line from lbw datasets

    Returns:
        image_path1: The path for the first image
        image_path2: The path for the second image


    """
    split_line = line.split()
    folder_name = split_line[0]
    if len(split_line) == 3:
        num_zeros1 = 4 - len(split_line[1])
        num_zeros2 = 4 - len(split_line[2])
    else:
        num_zeros1 = 4 - len(split_line[1])
        num_zeros2 = 4 - len(split_line[3])

    image_name1 = split_line[0].zfill(num_zeros1)
    image_name2 = split_line[0].zfill(num_zeros2)
    image_path1 = PATH_TO_DATASET + folder_name + image_name1
    image_path2 = PATH_TO_DATASET + folder_name + image_name2

    return image_path1, image_path2


def load_all_image_paths_siamese(pairs_file_path: str) -> List[Tuple[str, str]]:
    """Loads all image paths to ram

    Args:
        pairs_file_path: Path where the pairs are

    Returns:
        all_image_paths: All paths from the dataset

    """
    all_image_paths = []
    with open(pairs_file_path) as file:
        next(file)
        for raw_line in file:
            img_paths_tuple = process_raw_line_siamese(raw_line)
            all_image_paths.append(img_paths_tuple)

    return all_image_paths


def load_batch_of_images_siamese(
    image_paths: List[Tuple[str, str]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Given a list of image paths it loads a batch

    Args:
        image_paths: A batch size list of image paths

    Returns:
        batch_images: A batch of images

    """
    batch_images1 = np.zeros((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3))
    batch_images2 = np.zeros((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3))
    for index, paths_tuple in enumerate(image_paths):
        batch_images1[index] = cv2.imread(paths_tuple[0])
        batch_images2[index] = cv2.imread(paths_tuple[1])

    return batch_images1, batch_images2


def load_all_image_paths_convnet(data_dir: str) -> List[str]:
    """Loads all image paths for the convnet training

    Args:
        data_dir: Path where the data dir is

    Returns:
        image_paths: All image paths

    """
    img_paths = []
    for root, directories, filenames in os.walk(data_dir):
        for filename in filenames:
            extension = os.path.splitext(filename)[1]
            if extension in IMG_EXTENSIONS:
                file_path = os.path.join(root, filename)
                img_paths.append(file_path)

    return img_paths


def load_batch_of_data_convnet(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Given a list of image paths it loads a batch

    Args:
        image_paths: A batch size list of image paths

    Returns:
        batch_images: A batch of images

    """
    size_of_batch = len(image_paths)
    batch_images = np.zeros((size_of_batch, IMG_WIDTH, IMG_HEIGHT, 3))
    batch_labels = np.zeros((size_of_batch))
    for index, path in enumerate(image_paths):
        batch_images[index] = cv2.resize(cv2.imread(path), (150, 150))
        if LABEL in path:
            batch_labels[index] = 1

    return batch_images, batch_labels
