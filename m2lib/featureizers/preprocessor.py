import spacy
import nltk
from m2lib.readers.readdata import Read
from m2lib.pickler.picklable import Picklable, PickleDef
import pandas as pd
from gensim.parsing.preprocessing import preprocess_documents, remove_stopwords, preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text
from nltk.stem.wordnet import WordNetLemmatizer


class Pipeline(object):
    __instance = None
    steps = {}
    def __new__(cls):
        if Pipeline.__instance is None:
            Pipeline.__instance = object.__new__(cls)
        return Pipeline.__instance

    def __call__(cls, step):
        Pipeline.__instance.steps[f'{step.__name__}'] = step

class Preprocessor(Picklable, object):
    """
    This class will take in some documents like pandas series
    and turn into a dataframe with tokenized and all that good stuff
    dataframe.
    :param: documents
    :return: series and array
    """
    # __instance = None
    def __init__(self):

        self.original_doc = None
        self.pipeline_steps = None
        # self.data_store_kwargs =
        self.corpus_processed = None #self.pipeline(corpus)
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super(Preprocessor, self).__init__(**self.pickle_kwargs)

    # def __new__(cls):
    #     if Preprocessor.__instance is None:
    #         Preprocessor.__instance = object.__new__(cls)
    #     return Preprocessor.__instance


    def pipeline(self, corpus, dave_pipeline=False):
        # add first features
        print('running rudimentary preprocessing pipeline')
        self.pipeline_steps = [self.tokenize_gensim_string, self.lemmatize]
        if dave_pipeline:
            pass
        else:
            #pseudo pipeline for working on the documents
            corpus_piped = []
            for doc in corpus:
                modified_doc = doc
                for step in self.pipeline_steps:
                    # call step with kwargs and retain modifications
                    modified_doc = step(modified_doc)
                corpus_piped.append(modified_doc)
            self.corpus_processed = corpus_piped
            print(f'pickling preprocessor after pipeline run')
            self.save()
            return self.corpus_processed

    # decorator to add steps to the pipeline on class init
    def __make_dataframe(self):
        df = pd.DataFrame(self.feature_map)
        self.df = df

    def __add_feature(self, feature):
        self.feature_map[feature['name']] = feature['values']

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

    def __save_data_file(self):
        pass

    def save(self, obj=None):
        super(Preprocessor, self).save()
        pass

    def load(self):
        super(Preprocessor, self).load()



if __name__ == '__main__':

    read = Read(file='WikiLarge_Train.csv')
    train_set = read.read_dfs['WikiLarge_Train.csv'][:20]
    preprocessor = Preprocessor()
    processed = preprocessor.pipeline(train_set)
    print(processed)
    preprocessor.save()
    # preprocessor.pipeline(train_set['original_text'], dave_pipeline=False)
    print(preprocessor.corpus_processed)
