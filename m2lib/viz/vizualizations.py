from m2lib.model.gsdmmSttmModel import GSDMM, GSDMMModelStore
from m2lib.model.ldaModel import LDAModel, LDATFIDModel, LDATFIDModelStore, LDAModelStore
from m2lib.pickler.picklable import Picklable, PickleDef
from m2lib.featureizers.preprocessor import Preprocessor, PreprocessorStore
from m2lib.readers.readdata import Read
from m2lib.featureizers.bowfeature import BOWFeature, BOWFeatureStore
import pyLDAvis
import pyLDAvis.gensim


class PyLDAVizStore(Picklable):
    # can append to store values if needed
    def __init__(self, force_build=False):
        self.viz = []
        self.html = []
        self.K = []
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(force_build, **self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        super().load()


class PyLDAViz(Picklable):
    # only generates one instance and does not store values as pickle
    def __init__(self, force_build=False):
        self.viz = None
        self.html = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(force_build, **self.pickle_kwargs)

    def pipeline(self, model, corpus, dictionary):
        viz = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        html = viz.save_html()
        self.viz = viz
        self.html = html


    def save(self):
        super().save()

    def load(self):
        super().load()


if __name__ == '__main__':
    ldaStore = LDAModelStore()
    bowStore = BOWFeatureStore()
    pyviz = PyLDAViz()
    pyvizStore = PyLDAVizStore()
    pyviz.pipeline(ldaStore.model, bowStore.corpus_, bowStore.dictionary)

    pyvizStore.viz.append(pyviz.viz)
    pyvizStore.html.append(pyviz.html)
    pyvizStore.K.append(10)
    pyvizStore.save()



