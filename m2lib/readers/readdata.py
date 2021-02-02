import pandas as pd
import os
from configurations import RAW_DATA_DIR, DATA_DIR
from m2lib.pickler.picklable import Picklable, PickleDef


class ReadError(Exception):
    """
    Read Error Class might be useful somewhere
    """

    def __init__(self, file, ext=None, message='Encountered a read error'):
        self.file = file
        self.message = message
        self.ext = ext
        super().__init__(self.message)

    def __str__(self):
        pass

    def __call__(self):
        return f'''
        Error reading file {self.file}
        Error was: {self.message}
        '''


class ReadStore(Picklable):
    def __init__(self):
        self.dfs = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(**self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        super().load()


class Read(Picklable):
    """
    parameters: file or directory or list of files
    """

    def __init__(self, file=None):
        # defines allowed extensions
        self.allowed_extensions = ['txt', 'csv']
        self.data_dir = DATA_DIR
        self.read_dfs = {}
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super(Read, self).__init__(**self.pickle_kwargs)
        # logic to read in new files
        if file:
            if type(file) == list:
                for f in file:
                    if f in self.read_dfs:
                        print(f'{f} already has been read \n skipping {f}')
                        pass
                    else:
                        self.__readFilesDataFrames(file)
            else:
                if file in self.read_dfs:
                    print(f'{file} already has been read \n using loaded object')
                else:
                    self.__readFilesDataFrames(file)

    def __getExtension(self, file):
        return file.split('.')[-1]

    def __readToDataFrame(self, file):
        ext = self.__getExtension(file)
        if ext in self.allowed_extensions:
            try:
                df = pd.read_csv(os.path.join(self.data_dir, file))
                self.read_dfs[file] = df
                return df
            except Exception as e:
                print(e)
        else:
            raise ReadError(file, message='extension not in allowed extensions')

    def __readFilesDataFrames(self, file):
        if type(file) == list:
            for f in file:
                self.__readToDataFrame(f)
        else:
            if (type(file) == str):
                self.__readToDataFrame(file)
        # saves self to pickle file after reading frame or frames
        self.save()

    def add_file(self, file):
        """
        Can add a single file or multiple files
        params: list or str (file paths)
        updates read_dfs dict
        """
        f = None
        if type(file) == list:
            f = file
        else:
            f = [file]
        assert f, 'f not defined'
        self.__readFilesDataFrames(f)
        self.save()

    def add_dataframe(self, file):
        return self.__readToDataFrame(file)

    def save(self, obj=None):
        super(Read, self).save()
        pass

    def load(self):
        super(Read, self).load()


if __name__ == '__main__':
    reader = Read()
    print(reader.read_dfs)
    readStore = ReadStore()
    readStore.dfs = reader.read_dfs
    readStore.save()
