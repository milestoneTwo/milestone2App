from m2lib.pickler.pickler import Pickler
from m2lib.readers.readdata import Read
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Files to read from original data source')
    parser.add_argument("-files", nargs='+', help="add space delimited files needs one if specified", type=str)
    # parser.add_argument("-rerun", help="does poop", type=str)
    args = parser.parse_args()
    reader = Read(files=args.files)
    pickler = Pickler()
    pickler.add_pickle('data', 'datatest.pkl', reader)
