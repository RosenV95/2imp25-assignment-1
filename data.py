import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from typing import List

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


class Data:
    def __init__(self, low_path, high_path):
        low_df = self.read_data(low_path)
        high_df = self.read_data(high_path)

        low_reqs = list(low_df.text.values)
        high_reqs = list(high_df.text.values)

        low_reqs = self.preprocess(low_reqs)
        high_reqs = self.preprocess(high_reqs)


    def preprocess(self, reqs):
        reqs = self.tokenize(reqs)
        reqs = self.remove_capitals(reqs)
        reqs = self.remove_stopwords(reqs)
        reqs = self.stem(reqs)
        return reqs

    def tokenize(self, reqs):
        new_reqs = []
        for req in reqs:
            new_reqs.append(word_tokenize(req))
        return new_reqs

    def remove_capitals(self, reqs):
        new_reqs = []
        for req in reqs:
            new_req = []
            for word in req:
                new_word = word.lower()
                new_req.append(new_word)
            new_reqs.append(new_req)

        return new_reqs

    def remove_stopwords(self, reqs):
        new_reqs = []
        set_of_stopwords = set(stopwords.words('english'))
        for req in reqs:
            new_req = []
            for word in req:
                if word not in set_of_stopwords and len(word) > 1:
                    new_req.append(word)
            new_reqs.append(new_req)
        return new_reqs

    def stem(self, reqs):
        new_reqs = []
        for req in reqs:
            new_req = []
            for word in req:
                new_req.append(ps.stem(word))
            new_reqs.append(new_req)
        return new_reqs


    def read_data(self, path):
        df = pd.read_csv(path)
        return df
