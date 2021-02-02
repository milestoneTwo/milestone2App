from m2lib.featureizers.preprocessor import Preprocessor
from m2lib.pickler.picklable import Picklable, PickleDef
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases
from m2lib.readers.readdata import Read

class BOWFeature(Picklable):
    def __init__(self):
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        # make ngrams and add to corpus tokens
        self.dictionary = None
        self.corpus_ = None
        self.make_bow_args = {'no_below':3, 'no_above':0.5}
        super(BOWFeature, self).__init__(**self.pickle_kwargs)

    def pipeline(self, corpus, **kwargs):
        self.dictionary = Dictionary(corpus)
        manual_steps = [self.__make_bow_dictionary]
        corpus_ = []
        for doc in corpus:
            modified_doc = doc
            for step in manual_steps:
                modified_doc = step(modified_doc)
            corpus_.append(modified_doc)
        self.corpus_ = corpus_
        self.save()
        return self.corpus_

    def __make_bow_dictionary(self, doc):
        self.dictionary.filter_extremes(**self.make_bow_args)
        d2b = self.dictionary.doc2bow(doc)
        return d2b

    def set_make_bow_args(self, no_below, no_above):
        kwargs = {'no_above': no_above, 'no_below': no_below}
        for k, v in kwargs.items():
            if k in self.make_bow_args:
                self.make_bow_args[k] = v
        self.save()

    def save(self):
        super().save()

    def load(self):
        self.__dict__ = super(BOWFeature, self).load()

if __name__ == '__main__':
    read = Read(file='WikiLarge_Train.csv')
    train_set = read.read_dfs['WikiLarge_Train.csv']['original_text'][:20]
    preprocessor = Preprocessor()
    # preprocessor.train_phrase_model(train_set)
    processed = preprocessor.pipeline(train_set)
    bow = BOWFeature()
    bow.pipeline(preprocessor.corpus_)
    print(bow.corpus_)