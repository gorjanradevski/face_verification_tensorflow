from typing import List, Tuple
from utils.data_utils import process_raw_line
from utils.global_config import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH
import numpy as np
import cv2


def load_all_image_paths(pairs_file_path: str) -> List[Tuple[str, str]]:
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
            img_paths_tuple = process_raw_line(raw_line)
            all_image_paths.append(img_paths_tuple)

    return all_image_paths


def load_batch_of_images(
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
