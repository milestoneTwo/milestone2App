import pandas as pd
import numpy as np
import re
import statistics as stats
import pickle as pkl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from m2lib.pickler.picklable import Picklable, PickleDef
from configurations import PICKLE_DIR
import os

class Partb(Picklable):
    def __init__(self):
        self.df_train = None
        self.models = None
        pd = PickleDef(self)
        super().__init__(**pd())

    def pipeline(self, df):
        def get_mean_conc(text):
            sum_conc = 0
            word_count = max(len(text), 1)
            for word in text:
                try:
                    sum_conc += conc_dict[word]['Conc.M']
                except:
                    sum_conc += mean_conc

            return round(sum_conc / word_count, 4)

        def get_mean_aoa(text):
            sum_aoa = 0
            word_count = max(len(text), 1)
            for word in text:
                try:
                    sum_aoa += age_dict[word]['aoa']
                except:
                    sum_aoa += mean_aoa

            return round(sum_aoa / word_count, 4)

        def get_mean_perc(text):
            sum_perc = 0
            word_count = max(len(text), 1)
            for word in text:
                try:
                    sum_perc += age_dict[word]['perc_known']
                except:
                    try:
                        sum_perc += conc_dict[word]['Percent_known']
                    except:
                        sum_perc += mean_perc_known

            return round(sum_perc / word_count, 4)

        def get_mean_freq(text):
            list_freq = []
            for word in text:
                try:
                    list_freq.append(age_dict[word]['freq'])
                except:
                    list_freq.append(med_freq)

            if not list_freq:
                list_freq.append(0)

            return stats.median(list_freq)

        def count_non_basic(text):
            count = 0
            for word in text:
                try:
                    count += age_dict[word]['non_basic']
                except:
                    try:
                        count += conc_dict[word]['non_basic']
                    except:
                        count += 1
            return count

        def lem_combine(text_list):
            lem = WordNetLemmatizer()
            lem_word = []
            for word in text_list:
                lem_word.append(lem.lemmatize(word))

            return (' '.join(word for word in lem_word))
        df_train = df.copy()
        #step1
        df_train['clean_text'] = df_train['original_text'].apply(self.tokenize_and_remove_stops)
        df_train['syllables'] = df_train['clean_text'].apply(self.syllable_count)
        df_train['word_count'] = df_train['clean_text'].apply(lambda x: len(x))
        df_train['avg_syll_per_word'] = df_train['syllables'] / df_train['word_count']
        df_train['avg_syll_per_word'] = df_train['avg_syll_per_word'].fillna(0)

        #step2
        df_train['fc_ease'] = df_train.apply(self.flesch_kincaid_ease, axis=1)
        fc_mean = df_train['fc_ease'].dropna().mean()
        df_train['fc_ease'] = df_train['fc_ease'].fillna(fc_mean)

        #step3
        df_train.groupby('label')['fc_ease'].mean()

        #step4
        df_conc = pd.read_csv('Concreteness_ratings_Brysbaert_et_al_BRM.txt', sep='\t').fillna('0')
        basic_list = pd.read_csv('dale_chall.txt')['a'].to_list()

        df_conc['non_basic'] = df_conc['Word'].apply(lambda x: 0 if x in basic_list else 1)
        mean_conc = round(df_conc['Conc.M'].mean(), 4)
        print(df_conc.head(1))
        print(mean_conc)

        conc_keys = df_conc['Word'].tolist()

        dict_list = df_conc.set_index('Word').to_dict(orient='records')

        conc_dict = {}

        #step5
        df_age = pd.read_csv('AoA_51715_words.csv')
        df_age['non_basic'] = df_age['Lemma_highest_PoS'].apply(lambda x: 0 if x in basic_list else 1)
        # print(df_age.head())

        mean_aoa = round(df_age['AoA_Kup_lem'].mean(), 4)
        mean_perc_known = round(df_age['Perc_known_lem'].mean(), 4)
        med_freq = df_age['Freq_pm'].median()

        # print(mean_aoa, mean_perc_known, med_freq)

        age_keys = df_age['Word'].to_list()
        age_alt_keys = df_age['Alternative.spelling'].to_list()

        df_columns_for_dict = df_age[
            ['Freq_pm', 'Dom_PoS_SUBTLEX', 'AoA_Kup_lem', 'Perc_known_lem', 'non_basic']].copy()
        df_columns_for_dict.columns = ['freq', 'pos', 'aoa', 'perc_known', 'non_basic']
        dict_list = df_columns_for_dict.to_dict(orient='records')

        age_dict = {}
        for i in range(len(age_keys)):
            age_dict[age_keys[i]] = dict_list[i]

        for i in range(len(age_alt_keys)):
            if age_alt_keys[i] not in age_dict:
                age_dict[age_alt_keys[i]] = dict_list[i]

        for i in range(len(conc_keys)):
            conc_dict[conc_keys[i]] = dict_list[i]

        # step 6
        df_train['mean_conc'] = df_train['clean_text'].apply(get_mean_conc)
        df_train['mean_aoa'] = df_train['clean_text'].apply(get_mean_aoa)
        df_train['mean_perc_known'] = df_train['clean_text'].apply(get_mean_perc)
        df_train['mean_freq'] = df_train['clean_text'].apply(get_mean_freq)
        df_train['non_basic_words'] = df_train['clean_text'].apply(count_non_basic)
        df_train = df_train.fillna(0)

        # step 7
        df_train['lem_text'] = df_train['clean_text'].apply(self.lem_combine)

        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=30000)

        X_vec = vectorizer.fit_transform(df_train['lem_text'])
        y_vec = df_train['label']

        log_vec = LogisticRegression(max_iter=1000)
        log_vec.fit(X_vec, y_vec)

        df_train['logreg_prob'] = [num[0] for num in log_vec.predict_proba(X_vec)]

        # step 8
        X_train = df_train[
            ['word_count', 'avg_syll_per_word', 'fc_ease', 'mean_conc', 'mean_aoa', 'mean_perc_known', 'mean_freq',
             'non_basic_words', 'logreg_prob']]
        y_train = df_train['label']

        # step 9
        rf_clf = RandomForestClassifier(max_depth=20, min_samples_leaf=1, n_estimators=500)
        rf_clf.fit(X_train, y_train)

        models = [log_vec, rf_clf]
        self.models = models

        with open(os.path.join(PICKLE_DIR, 'log_reg.pkl'), 'wb') as handle:
            pkl.dump(log_vec, handle, protocol=pkl.HIGHEST_PROTOCOL)

        with open(os.path.join(PICKLE_DIR, 'rand_forest.pkl'), 'wb') as handle:
            pkl.dump(rf_clf, handle, protocol=pkl.HIGHEST_PROTOCOL)

        self.df_train = df_train
        # df_train.head()

    def tokenize_and_remove_stops(self, text):
        text_list = word_tokenize(text)
        stop_word_set = set(stopwords.words('english'))
        clean_list = [word.lower() for word in text_list if word.lower() not in stop_word_set]
        clean_list = [word for word in clean_list if re.match('^[a-z]+$', word)]

        return clean_list

    def syllable_count(self, clean_text_list):
        count = 0
        for word in clean_text_list:
            word = word.lower()
            vowels = "aeiouy"
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith("e"):
                count -= 1
            if count == 0:
                count += 1
        return count

    def flesch_kincaid_ease(self, row):
        return round(206.835 - 1.015 * (row['word_count']) - 84.6 * (row['avg_syll_per_word']), 2)
