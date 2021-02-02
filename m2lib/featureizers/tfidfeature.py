from m2lib.featureizers.preprocessor import Preprocessor
from m2lib.pickler.picklable import Picklable, PickleDef
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases
from m2lib.readers.readdata import Read
from gensim.models import TfidfModel
from m2lib.featureizers.bowfeature import BOWFeature
from gensim.corpora import Dictionary

class TFIDFeatureStore(Picklable):
    def __init__(self, force_build=False):
        self.corpus_ = None
        self.dictionary = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(force_build, **self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        super().load()

class TFIDFeature(Picklable):
    def __init__(self, force_build=False):
        self.corpus_ = None
        self.tfid_args = {
        }
        self.dictionary = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super(TFIDFeature, self).__init__(force_build, **self.pickle_kwargs)

    def pipeline(self, bow, dictionary):
        if super().check_if_force_build():
            model = TfidfModel(bow)
            self.corpus_ = model[bow]
            self.dictionary = dictionary
            self.save()
        else:
            pass

    def save(self, obj=None):
        super(TFIDFeature, self).save()

    def load(self):
        super(TFIDFeature, self).load()

if __name__ == '__main__':
    read = Read(file='WikiLarge_Train.csv')
    train_set = read.read_dfs['WikiLarge_Train.csv']['original_text'][:20]
    preprocessor = Preprocessor()
    # preprocessor.train_phrase_model(train_set)
    processed = preprocessor.pipeline(train_set)
    bow = BOWFeature()
    bow.pipeline(preprocessor.corpus_)
    tfidfeature = TFIDFeature()
    tfidfeature.pipeline(bow.corpus_, bow.dictionary)
    tfidvec = tfidfeature.corpus_

