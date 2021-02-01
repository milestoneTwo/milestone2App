import pandas as pd
import os
from configurations import RAW_DATA_DIR

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


class Read():
    """
    parameters: file or directory
    """
    def __init__(self, file=None, files=None):
        self.file = file
        self.files = files
        # defines allowed extensions
        self.allowed_extensions = ['txt', 'csv']
        self.raw_data_dir = RAW_DATA_DIR
        self.read_dfs = {}
        self.__readMultipleFilesDataFrames(self.files)

    def __getExtension(self, file):
        return file.split('.')[-1]

    def __readToDataFrame(self, file):
        ext = self.__getExtension(file)
        if ext in self.allowed_extensions:
            df = pd.read_csv(os.path.join(self.raw_data_dir,file))
        else:
            raise ReadError(file, message='extension not in allowed extensions')
        return df

    def __readMultipleFilesDataFrames(self, files):
        print('reading several defined files to a dataframe')
        def update_files(files):
            for file in files:
                try:
                    df = self.__readToDataFrame(file)
                    self.read_dfs[file] = df
                except Exception as e:
                    print(e)

        update_files(files)


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
        self.__readMultipleFilesDataFrames(f)

    def __pickle(self):
        pass

    def __repr__(self):
        pass

    def __call__(self):
        pass
