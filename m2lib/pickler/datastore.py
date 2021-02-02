from m2lib.pickler.picklable import Picklable, PickleDef

class DataStoreDef:
    def __init__(self, obj):
        print(obj.__name__)

    def __call__(self):
        return self.__dict__

class DataStore(Picklable):
    def __init__(self, series):
        self.obj = series
        dd = DataStoreDef(series)
        super(DataStore, self).__init__(**dd())

    def save(self):
        super().save(self.obj)

    def load(self):
        self.__dict__ = super(DataStore, self).load()

if __name__ == '__main__':
    my_list = [1,2,3,4]
    ds = DataStore(my_list)