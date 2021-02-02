from m2lib.pickler.picklable import Picklable, PickleDef
from gensim.models import LdaModel
from m2lib.featureizers.preprocessor import Preprocessor
from m2lib.readers.readdata import Read
from m2lib.featureizers.bowfeature import BOWFeature


class LDAModel(Picklable):
    def __init__(self):
        self.model = None
        self.lda_args = {
            'chunksize': 2000,
            'alpha' : 'auto',
            'eta': 'auto',
            'iterations' : 400,
            'num_topics' : 10,
            'passes' : 20,
            'eval_every' : None
        }
        pd = PickleDef(self)
        super(LDAModel, self).__init__(**pd())

    def __call__(self, *args, **kwargs):
        pass


    def train_model(self, corpus, dictionary):
        print('training LDA model')

        temp = dictionary[0]
        id2word = dictionary.id2token

        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            **self.lda_args
        )
        self.model = model
        self.save()

    def set_lda_args(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.lda_args:
                self.lda_args[k] = v
            else:
                raise KeyError

    def save(self, obj=None):
        super(LDAModel, self).save()
        pass

    def load(self):
        super(LDAModel, self).load()

#TODO Took come shortcuts on inheritance here
class LDATFIDModel(LDAModel, Picklable):
    def __init__(self):
        LDAModel.__init__(self)
        pd = PickleDef(self)
        Picklable.__init__(self, **pd())

    def train_model(self, corpus, dictionary):
        super().train_model(corpus, dictionary)

    def set_lda_args(self, **kwargs):
        super().set_lda_args(**kwargs)

    def save(self, obj=None):
        super().save()
        pass

    def load(self):
        super().load()


if __name__ == '__main__':
    read = Read(file='WikiLarge_Train.csv')
    train_set = read.read_dfs['WikiLarge_Train.csv']['original_text'][:20]
    preprocessor = Preprocessor()
    # preprocessor.train_phrase_model(train_set)
    processed = preprocessor.pipeline(train_set)
    lda_model = LDATFIDModel()
    bow = BOWFeature()
    bow.pipeline(preprocessor.corpus_)
    lda_model.train_model(bow.corpus_, bow.dictionary)
    print(lda_model.model)
