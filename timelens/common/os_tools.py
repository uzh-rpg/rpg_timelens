import glob
import os
import re

import numpy as np

def make_glob_filename_iterator(file_template):
    return sorted(glob.glob(file_template))

def find_leaf_folders(root_folder):
    """Returns list of leaf folders. 
    
    "Leaf" folder is non-hidden folder that does 
    not have non-hidden subfolders. 
    """
    leaf_folders = []
    for sequence_folder, child_folders, _ in os.walk(root_folder):
        # Skip hidden directories.
        if os.path.basename(sequence_folder)[0] == ".":
            continue
        # Remove hidden directories.
        child_folders = [
            child_folder for child_folder in child_folders if child_folder[0] != "."
        ]
        if not child_folders:
            leaf_folders.append(sequence_folder)
    return leaf_folders


def find_files_by_template(folder, file_template, is_recursive=False):
    """Returns list of files matching the template.
    
    The search is performed recursively in all subfolders.
    """
    if is_recursive:
        return sorted(glob.glob(os.path.join(folder, '**', file_template), recursive=True))
    return sorted(glob.glob(os.path.join(folder, file_template)))


def make_filename_iterator(filename_template):
    """Returns iterator over filenames.
    
    Args:
        filename_template: full path to the folder and filename template,
                           e.g. '/path/to/file/{:d}.npz'. 
    """
    index = 0
    filename = filename_template.format(index)
    while os.path.isfile(filename):
        yield filename
        index += 1
        filename = filename_template.format(index)


def list_to_file(filename, lst):
    """Saves every item of the list as a line in the file."""
    lst = [item.rstrip("\n") + "\n" for item in lst[:-1]] + [lst[-1].rstrip("\n")]
    with open(filename, "w") as f:
        f.writelines(lst)


def file_to_list(filename):
    """Adds every line of the file to list."""
    with open(filename) as f:
        examples_list = [line.rstrip("\n") for line in f.readlines()]
    return examples_list
