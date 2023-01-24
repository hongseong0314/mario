import os

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)