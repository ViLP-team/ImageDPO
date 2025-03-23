import glob
import os


def is_folder_empty(folder_path, check_folders_postfix="_groundingdino"):
    """
    Check if a folder is empty.
    We do this checking by see if the folder contains any sub-fodler ending with check_folders_postfix.
    """
    if len(glob.glob(os.path.join(folder_path, f"*{check_folders_postfix}"))) > 0:
        return False
    else:
        return True
