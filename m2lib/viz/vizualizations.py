from m2lib.model.gsdmmSttmModel import GSDMM, GSDMMModelStore
from m2lib.model.ldaModel import LDAModel, LDATFIDModel, LDATFIDModelStore, LDAModelStore
from m2lib.pickler.picklable import Picklable, PickleDef
from m2lib.featureizers.preprocessor import Preprocessor, PreprocessorStore
from m2lib.readers.readdata import Read
from m2lib.featureizers.bowfeature import BOWFeature, BOWFeatureStore

class PyLDAVizStore(Picklable):
    def __init__(self, force_build=False):
        self.viz = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(force_build, **self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        super().load()


class PyLDA(Picklable):
    def __init__(self, force_build=False):
        self.viz = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(force_build, **self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        super().load()


if __name__ == '__main__':
    ps = PreprocessorStore()
    bowStore = BOWFeatureStore()


