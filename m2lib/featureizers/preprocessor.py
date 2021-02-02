import spacy
import nltk
from m2lib.readers.readdata import Read
from m2lib.pickler.picklable import Picklable, PickleDef
from m2lib.pipelines.pipeline import Pipeline
import pandas as pd
from gensim.parsing.preprocessing import preprocess_documents, remove_stopwords, preprocess_string, strip_tags, \
    strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.phrases import Phrases
from nltk import ngrams
from tqdm import tqdm


class PhraseModel(Picklable):
    def __init__(self):
        self.phrase_model = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super(PhraseModel, self).__init__(**self.pickle_kwargs)

    def train_phrase_model(self, sentences, force=False, **kwargs):
        def train(sentences, **kwargs):
            phrases = Phrases(sentences, min_count=1, threshold=0.1)
            return phrases

        print('checking if phrase model already exists')
        if self.phrase_model:
            print('phrase model is present')
            if force:
                print('but we want to force update')
                print('training phrase model one does not exist')
                self.phrase_model = train(sentences, **kwargs)
                self.save()
            else:
                print('phrase model exists returning to you')
                return self.phrase_model
        else:
            self.phrase_model = train(sentences, **kwargs)
            self.save()

    def save(self, obj=None):
        super(PhraseModel, self).save()
        pass

    def load(self):
        super(PhraseModel, self).load()

class PreprocessorStore(Picklable):
    def __init__(self):
        self.corpus_ = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(**self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        super().load()


class Preprocessor(Picklable):
    """
    This class will take in some documents like pandas series
    and turn into a dataframe with tokenized and all that good stuff
    dataframe.
    :param: documents
    :return: series and array
    """

    def __init__(self, force_build=False):
        self.pipeline_steps = None
        self.corpus_ = None
        self.phrases = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super(Preprocessor, self).__init__(force_build, **self.pickle_kwargs)

    def pipeline(self, corpus, dave_pipeline=False):
        manual_steps = [self.tokenize_gensim_string, self.lemmatize, self.lower, self.make_ngrams]
        # self.train_phrase_model(corpus)
        if super().check_if_force_build():
            print('running rudimentary preprocessing pipeline')
            # pseudo pipeline for working on the documents
            corpus_ = []
            for doc in tqdm(corpus):
                modified_doc = doc
                for step in manual_steps:
                    # call step with kwargs and retain modifications
                    modified_doc = step(modified_doc)
                corpus_.append(modified_doc)
            self.corpus_ = corpus_
            print(f'pickling preprocessor after pipeline run')
            self.save()
            return self.corpus_
        else:
            pass

    def step(function):
        pipeline = Pipeline()
        pipeline(function)

    def __make_dataframe(self):
        df = pd.DataFrame(self.feature_map)
        self.df = df

    def __add_feature(self, feature):
        self.feature_map[feature['name']] = feature['values']

    def train_phrase_model(self, corpus, force=False):
        # train the ngram -> phrase model or get pickle implements picklable
        self.phrases = PhraseModel().train_phrase_model(corpus, force=force)

    def tokenize_gensim_string(self, doc):
        CUSTOM_FILTERS = [
            strip_tags,
            strip_punctuation,
            strip_multiple_whitespaces,
            strip_numeric,
            remove_stopwords,
            # strip_short,
            # stem_text
        ]
        doc_ = preprocess_string(doc, CUSTOM_FILTERS)
        return doc_

    def lemmatize(self, doc):
        lemmatizer = WordNetLemmatizer()
        doc_ = [lemmatizer.lemmatize(token) for token in doc]
        return doc_

    def lower(self, doc):
        return [t.lower() for t in doc]

    def make_ngrams(self, doc, n=2):
        doc_ = doc + ['_'.join(gram) for gram in ngrams(doc, n)]
        return doc_

    def __save_data_file(self):
        pass

    def save(self, obj=None):
        super(Preprocessor, self).save()
        pass

    def load(self):
        super(Preprocessor, self).load()


if __name__ == '__main__':
    read = Read(file='WikiLarge_Train.csv')
    train_set = read.read_dfs['WikiLarge_Train.csv']['original_text']
    preprocessor = Preprocessor(force_build=False)
    # preprocessor.train_phrase_model(train_set)
    processed = preprocessor.pipeline(train_set)
    # preprocessor.pipeline(train_set['original_text'], dave_pipeline=False)
    ps = PreprocessorStore()
    ps.corpus_ = preprocessor.corpus_
    ps.save()
