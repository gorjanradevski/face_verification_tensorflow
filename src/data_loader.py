from typing import List, Tuple
from utils.data_utils import process_raw_line

def load_all_image_paths(pairs_file_path: str) -> List[Tuple[str,str]]:
    all_image_paths = []
    with open(pairs_file_path) as file:
        next(file)
        for raw_line in file:
            img_paths_tuple = process_raw_line(raw_line)
            all_image_paths.append(img_paths_tuple)

    return all_image_paths