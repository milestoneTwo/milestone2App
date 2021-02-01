import os
from pathlib import Path

# current directory
CURR_DIR = Path(__file__).resolve()

# project root directory
ROOT_DIR = CURR_DIR.parent

# subdirectories
DATA_DIR = os.path.join(ROOT_DIR, 'data/')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw/')
PICKLE_DIR = os.path.join(ROOT_DIR, 'pickles/')

# files
ZIP_DATA = os.path.join(ROOT_DIR, 'umich-siads-695-predicting-text-difficulty.zip')
TRAIN_DATA = 'WikiLarge_Train.csv'
TEST_DATA = 'WikiLarge_Test.csv'
DATA_FILES = ['WikiLarge_Test.csv', 'WikiLarge_Train.csv']

if __name__ == '__main__':
    print('nothing done here in this file... yet')
    pass


