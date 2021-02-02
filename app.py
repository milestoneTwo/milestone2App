import sys
import argparse
import nltk
import os
from m2lib.readers.readdata import Read
from m2lib.featureizers.preprocessor import Preprocessor
from m2lib.featureizers.bowfeature import BOWFeature
from m2lib.featureizers.tfidfeature import TFIDFeature
from m2lib.model.ldaModel import LDAModel, LDATFIDModel
import configurations
from zipfile import ZipFile
from m2lib.model.gsdmmSttmModel import GSDMM
import time
import webbrowser

toy = False

def heading(process):
    print('\n')
    print('------------------------------------------------------------')
    print(f'{process}')
    print('------------------------------------------------------------')
    print('\n')

def subheading(process):
    print(f'*****{process}*****')
    print('\n')

def start():
    print("""
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMM             MMMMMMMMMMMMMMMMM             MMMMMMMMM
    MMMMMMMM              MMMMMMMMMMMMMMM              MMMMMMMMM
    MMMMMMMM                MMMMMMMMMMM                MMMMMMMMM
    MMMMMMMM                 MMMMMMMMM                 MMMMMMMMM
    MMMMMMMM                  MMMMMMM                  MMMMMMMMM
    MMMMMMMMMMMM               MMMMM                MMMMMMMMMMMM
    MMMMMMMMMMMM                MMM                 MMMMMMMMMMMM
    MMMMMMMMMMMM                 V                  MMMMMMMMMMMM
    MMMMMMMMMMMM                                    MMMMMMMMMMMM
    MMMMMMMMMMMM         ^               ^          MMMMMMMMMMMM
    MMMMMMMMMMMM         MM             MM          MMMMMMMMMMMM
    MMMMMMMMMMMM         MMMM         MMMM          MMMMMMMMMMMM
    MMMMMMMMMMMM         MMMMM       MMMMM          MMMMMMMMMMMM
    MMMMMMMMMMMM         MMMMMM     MMMMMM          MMMMMMMMMMMM
    MMMMMMMM                MMMM   MMMM                MMMMMMMMM
    MMMMMMMM                MMMMMVMMMMM                MMMMMMMMM
    MMMMMMMM                MMMMMMMMMMM                MMMMMMMMM
    MMMMMMMM                MMMMMMMMMMM                MMMMMMMMM
    MMMMMMMM                MMMMMMMMMMM                MMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    """)

def build_app(curr_dir):
    start()
    time.sleep(2)
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


    """Feature Building """
    # Developing Features
    heading('Building Features')
    subheading('Building BOW')

    # pass in the preprocessed corpus
    bow = BOWFeature()
    bow.pipeline(preprocessor.corpus_)
    feature_bow = bow.corpus_
    dictionary_bow = bow.dictionary

    subheading('Building TFID')
    tfidfeature = TFIDFeature()
    tfidfeature.pipeline(bow.corpus_, bow.dictionary)


    """Model Building Aread"""

    heading('Building Models')
    subheading('Building LDA with BOW')
    lda = LDAModel()
    lda.train_model(feature_bow, dictionary_bow)

    heading('Building LDA with TFID')
    lda_tfid = LDATFIDModel()
    lda_tfid.train_model(tfidfeature.corpus_, tfidfeature.dictionary)

    # GSDMM MovieProcessGroup STTM Model
    heading('Building GSDMM MovieGroupProcess Model')
    gsdmm = GSDMM()
    gsdmm.train_model(preprocessor.corpus_)

def partb_pipeline():


def run_server():
    os.system('cd Milestone2Docs && mkdocs serve')
    webbrowser.open('127.0.0.1:8000')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run build_app to read data files and generate main app objects", nargs=1)
    parser.add_argument("--run", help="run build_app to read data files and generate main app objects", nargs=1)
    args = parser.parse_args()
    if args.run[0] == 'build_app':
        build_app(os.getcwd())
