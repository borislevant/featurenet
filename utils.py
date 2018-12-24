import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_files_from_folder(directory):
    files = [os.path.abspath(os.path.join(directory, x)) for x in os.listdir(directory)]
    return files
