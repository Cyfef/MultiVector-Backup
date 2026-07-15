import os
import torch
from pathlib import Path
import numpy as np

from typing import List, Any, Tuple


def sort_by_indices(indices: List[int], embeddings: List[Any]) -> Tuple[List[int], List[Any]]:
    if len(indices) != len(embeddings):
        raise ValueError("Length of indices and embeddings must be the same")
    
    int_indices = [int(idx) for idx in indices]
    # Create a list of tuples (index, embedding) and sort by index
    paired = list(zip(int_indices, embeddings))
    paired.sort(key=lambda x: x[0])
    
    # Unzip the sorted pairs back into two lists
    sorted_indices, sorted_embeddings = zip(*paired)
    
    return list(sorted_indices), list(sorted_embeddings)


def sort_by_indices_multiple(indices: List[str], *other_lists) -> Tuple[List[str], ...]:
    
    if not other_lists:
        raise ValueError("At least one other list must be provided")
    
    # Check if all lists have the same length
    all_lists = [indices] + list(other_lists)
    if not all(len(lst) == len(indices) for lst in other_lists):
        raise ValueError("All lists must have the same length")
    
    # Convert string indices to integers for sorting, but keep original strings
    # Create a list of tuples (int_index, original_index, *other_values) and sort by int_index
    paired = [(int(idx), idx, *other_values) for idx, *other_values in zip(indices, *other_lists)]
    paired.sort(key=lambda x: x[0])  # Sort by the integer version of the index
    
    # Unzip the sorted pairs back into separate lists
    sorted_lists = list(zip(*paired))
    
    # Return sorted lists (skip the first element which is the int index used for sorting)
    return tuple(lst for lst in sorted_lists)


# Directory containing the .pt files
pt_dir = "/data1/coco/coco2017/embeddings_2_4/"

# Get all .pt files in the directory
pt_files = list(Path(pt_dir).glob("*.pt"))

print(f"Found {len(pt_files)} .pt files")


# Read each .pt file
image_embeddings = []
image_indices = []
image_embeddings_patch_2 = []
image_embeddings_patch_4 = []
count = 0
for pt_file in pt_files:
    count += 1
    if count % 1000 == 0:
        print(f"Processed {count} files")
    data = torch.load(pt_file)
    image_indices.extend(data['cached_img_ls'])
    image_embeddings.extend(data['img_emb'])
    image_embeddings_patch_2.extend(data['patch_emb_ls'][0])
    image_embeddings_patch_4.extend(data['patch_emb_ls'][1])

all_list = sort_by_indices_multiple(image_indices, image_embeddings, image_embeddings_patch_2, image_embeddings_patch_4)

# Directory containing the .pt files
caption_dir = "/data1/ali/mscocoresults/caption-embeddings/"

# Get all .pt files in the directory
caption_pt_files = list(Path(caption_dir).glob("*.pt"))

print(f"Found {len(caption_pt_files)} .pt files")


# Read each .pt file
caption_embeddings = []
caption_indices = []
count = 0
for pt_file in caption_pt_files:
    count += 1
    if count % 1000 == 0:
        print(f"Processed {count} files")
    data = torch.load(pt_file, weights_only=False)
    caption_embeddings.extend(data['embeddings'])
    caption_indices.extend(data['indices'])
    
caption_sorted_indices, caption_sorted_embeddings = sort_by_indices(caption_indices, caption_embeddings)


image_indices = all_list[0]
image_indices_txt = all_list[1]
image_embeddings = all_list[2]
image_embeddings_patch_2 = all_list[3]
image_embeddings_patch_4 = all_list[4]

print(len(image_indices), len(caption_sorted_indices))


# save image_indices, image_indices_txt, image_embeddings, image_embeddings_patch_2, image_embeddings_patch_4, caption_sorted_indices, caption_sorted_embeddings
from collections import Counter

def find_extra_item_with_index(list1, list2):
    """
    Finds the extra item and its index in the longer list.
    """
    # 1. Determine which list is longer
    if len(list1) > len(list2):
        longer_list = list1
        shorter_list = list2
    else:
        longer_list = list2
        shorter_list = list1

    # 2. Find the extra item using Counter
    counts_longer = Counter(longer_list)
    counts_shorter = Counter(shorter_list)
    
    # The difference will be a Counter with the single extra item
    diff = counts_longer - counts_shorter
    extra_item = list(diff.keys())[0]

    # 3. Count how many times the extra item appears in the shorter list
    count_in_shorter = shorter_list.count(extra_item)

    # 4. Find the index of the specific extra occurrence in the longer list
    seen_count = 0
    for i, item in enumerate(longer_list):
        if item == extra_item:
            # We've found an occurrence of the extra item.
            # Now we need to see if it's the one that is truly "extra".
            seen_count += 1
            if seen_count > count_in_shorter:
                # This is the first occurrence that doesn't have a matching counterpart
                # in the shorter list, so it must be the extra one.
                return extra_item, i
    
    return None, -1 # This case should not be reached if the inputs are valid


item, index  = find_extra_item_with_index(caption_sorted_indices, image_indices)
caption_sorted_indices.pop(index)
caption_sorted_embeddings.pop(index)
print("Extra Index removed Done")
