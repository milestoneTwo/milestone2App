import sys
import argparse
import nltk
import os
from m2lib.readers.readdata import Read
from m2lib.featureizers.preprocessor import Preprocessor
from m2lib.featureizers.bowfeature import BOWFeature
import configurations
from zipfile import ZipFile

toy = True

def heading(process):
    print('\n')
    print('------------------------------------------------------------')
    print(f'{process}')
    print('------------------------------------------------------------')
    print('\n')

def subheading(process):
    print(f'{process}')
    print('\n')

def build_app(curr_dir):
    # setup directories
    try:
        os.mkdir(configurations.PICKLE_DIR)
    except FileExistsError:
        pass

    try:
        os.mkdir(configurations.DATA_DIR)
    except FileExistsError:
        pass

    # unzip data
    with ZipFile(configurations.ZIP_DATA, 'r') as zip:
        zip.extractall(configurations.DATA_DIR)

    # download wordnet
    nltk.download('wordnet')

    heading('Reading Data')
    # read data into readers
    reader = Read([configurations.TRAIN_DATA, configurations.TEST_DATA])

    # Preprocess
    heading('Preprocessing Data')
    if toy:
        train_data = reader.read_dfs[configurations.TRAIN_DATA]['original_text'][:47]
    else:
        train_data = reader.read_dfs[configurations.TRAIN_DATA]['original_text'][:]

    preprocessor = Preprocessor()
    preprocessor.pipeline(train_data)

    # Developing Features
    heading('Building Features')
    subheading('Building BOW')
    # pass in the preprocessed corpus
    bow = BOWFeature(preprocessor.corpus_processed)
    bow.save()
    feature_bow = bow.data

    heading('Building LDA Model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run build_app to read data files and generate main app objects", nargs=1)
    args = parser.parse_args()
    if args.run[0] == 'build_app':
        build_app(os.getcwd())
