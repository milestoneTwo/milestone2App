from m2lib.featureizers.preprocessor import Preprocessor
from m2lib.pickler.picklable import Picklable, PickleDef
from gensim.corpora import Dictionary

class BOWFeature(Picklable):
    def __init__(self, corpus, ngrams=None):
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        assert corpus != None, 'corpus is None type must sort that out'
        # make ngrams and add to corpus tokens
        if ngrams:
            pass
        self.data_dictionary = Dictionary(corpus)
        super(BOWFeature, self).__init__(**self.pickle_kwargs)

    def make_bow_dictionary(self, corpus, **kwargs):
        self.data_dictionary.filter_extremes(no_below=20, no_above=0.5)
        data = [self.data_dictionary.doc2bow(doc) for doc in corpus]
        self.save()
        return data

    def save(self):
        super().save()

    def load(self):
        self.__dict__ = super(BOWFeature, self).load()