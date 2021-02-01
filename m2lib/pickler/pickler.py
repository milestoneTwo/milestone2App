import pickle
import os
from pathlib import Path
import json
from configurations import PICKLE_DIR


class PicklerError(Exception):
    """
    Pickle Error Class might be useful somewhere
    """
    def __init__(self, file, ext=None, message='Encountered a pickler error'):
        # self.file = file
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        pass

    def __call__(self):
        pass


class Pickler:
    """
    On Init looks for files in the pickles directory
    Saves file names and creates a pickles definition file
    This allows you to recall your object by name
    """
    def __init__(self):
        self.pickle_def_fname = 'pickledef.json'
        self.pickle_def_path = os.path.join(PICKLE_DIR, self.pickle_def_fname)
        if self.__def_exists():
            self.pickle_defs = self.__get_defs()
        else:
            self.pickle_defs = None
        # self.pickles stores pickles
        self.pickles = {}
        # has the names of the pickle files in the dir
        self.__pickle_files = self.__get_pickle_files()

    def __get_pickle_files(self):
        """
        gets a list of files in the pickles dir
        """
        filenames = []
        for file in os.listdir(PICKLE_DIR):
            if file.endswith('.pkl'):
                filenames.extend(file)
        # make some logic as this will not throw errors if files are empty
        if len(filenames) == 0:
            return None
        else:
            return filenames

    def __def_exists(self):
        """
        checks if there is a definition file in the pickles dir
        """
        def_file = Path(self.pickle_def_path)
        if def_file.exists():
            return True
        else:
            return False

    def __get_defs(self):
        """
        gets the definitions from the pickles files
        """
        if self.__def_exists():
            try:
                with open(self.pickle_def_path, 'r') as f:
                    defs_ = json.load(f)
                    self.pickle_defs = defs_
                    return defs_
            except Exception as e:
                print(e)

    def __dump_defs(self):
        """
        dumps pickle defs to a json file in the pickles dir
        """
        with open(self.pickle_def_path, 'w') as f:
            json.dump(self.pickle_defs, f, indent=4)

    def __save_pickle(self, name, fname, obj):
        file = os.path.join(PICKLE_DIR, fname)
        try:
            with open(file, 'wb') as f:
                pickle.dump(obj, f)
        except Exception as e:
            raise e

    def __update_def(self, pickle_obj):
        """
        if pickle definitions exists it updates the class attribute
        then will dump to file to save it

        else

        creates pickle definition dict attribute
        """
        if self.pickle_defs:
            self.pickle_defs[pickle_obj['name']] = pickle_obj['def_']
            self.__dump_defs()
        else:
            self.pickle_defs = {}
            self.pickle_defs[pickle_obj['name']] = pickle_obj['def_']
            self.__dump_defs()

    def __rollback_def(self, pickle_obj):
        if self.pickle_defs:
            del self.pickle_defs[pickle_obj['name']]
            self.__dump_defs()

    #TODO: make a namespace or class for defining a pickle
    def add_pickle(self, name, fname, obj):
        """
        provides method to add a pickle to the pickle objects
        """
        definition = {
            'name': name,
            'def_': {
                'fname': fname,
                'obj_type': str(type(obj))
            }
        }
        # update pickle def before writing to file
        self.__update_def(definition)
        # write pickle!
        try:
            self.__save_pickle(name, fname, obj)
        except Exception as e:
            # rollback def update if adding pickle fails
            print(e)
            print('rolling back pickle def update')
            self.__rollback_def(definition)

    # TODO: Rework getting all the pickles files
    def get_all_pickles(self):
        """
        loops through pickles files in pickles dir and gets them all
        returns a list of pickle objects
        """
        if self.__pickle_files:
            # self.pickles = {pickle.load(file) for file in self.__pickle_files]
            return self.pickles
        else:
            raise FileNotFoundError

    def check_pickle(self, **kwargs):
        """
        get a named pickle by name or by file name
        kwargs
        use fname
        name
        """
        if self.pickle_defs:
            if kwargs['name'] in self.pickle_defs:
                return True
            else:
                return False

    def get_pickle(self, **kwargs):
        if self.check_pickle(**kwargs):
            fname = os.path.join(PICKLE_DIR, kwargs['fname'])
            try:
                if os.path.getsize(fname) > 0:
                    with open(fname, 'rb') as f:
                        return pickle.load(file=f)
                else:
                    print('pickle file is empty')
                    return False
            except FileNotFoundError as e:
                print(f'pickle file not found for {kwargs["name"]} at {fname}')
                self.__rollback_def(**kwargs)
                return False
        else:
            print(f'Pickle not found by name {kwargs["name"]}')

    def __repr__(self):
        pass

    def __call__(self):
        pass

if __name__ == '__main__':
    pass
else:
    pass




