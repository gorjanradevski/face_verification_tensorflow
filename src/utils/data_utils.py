from typing import Tuple
from utils.global_config import PATH_TO_DATASET

def process_raw_line(line: str) -> Tuple[str, str]:
    """

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
