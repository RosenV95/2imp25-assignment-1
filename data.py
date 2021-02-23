import pandas as pd
import nltk
import math
from math import *
import numpy
import csv
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from typing import List
from scipy.spatial import distance
from decimal import Decimal
import textdistance
from scipy.stats import pearsonr, spearmanr



nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


class Data:
    def __init__(self, low_path, high_path, treshold, own_implementation):

        low_df = self.read_data(low_path)
        high_df = self.read_data(high_path)

        low_reqs = list(low_df.text.values)
        high_reqs = list(high_df.text.values)

        low_req_ids = list(low_df.id.values)
        high_req_ids = list(high_df.id.values)

        low_reqs = self.preprocess(low_reqs)
        high_reqs = self.preprocess(high_reqs)

        vocabulary = self.create_vocabulary(low_reqs, high_reqs)

        low_req_vocabulary = self.create_req_vocabulary(low_reqs)
        high_req_vocabulary = self.create_req_vocabulary(high_reqs)

        number_of_reqs = len(low_reqs) + len(high_reqs)

        low_req_vectors = self.create_vectors(vocabulary, low_reqs, low_req_vocabulary, number_of_reqs)
        high_req_vectors = self.create_vectors(vocabulary, high_reqs, high_req_vocabulary, number_of_reqs)


        if own_implementation == True:
            sim_matrix=self.create_pearson_matrix(low_req_vectors, high_req_vectors)
        else:
            sim_matrix = self.create_matrix(low_req_vectors, high_req_vectors)

        self.write_output(sim_matrix, treshold, low_req_ids, high_req_ids)




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

    def create_vocabulary(self, low_reqs, high_reqs):
        dict = {}
        for all_reqs in [low_reqs, high_reqs]:
            for req in all_reqs:
                req_dict = {}
                for word in req:
                    if word not in req_dict:
                        if word in dict:
                            dict[word] += 1
                            req_dict[word] = 1
                        else:
                            dict[word] = 1
                            req_dict[word] = 1

        vocabulary = list(dict.items())
        return vocabulary

    def create_req_vocabulary(self, reqs):
        vocabulary = []
        for req in reqs:
            dict = {}
            for word in req:
                if word in dict:
                    dict[word] += 1
                else:
                    dict[word] = 1
            vocabulary.append(dict)
        return vocabulary


    def create_vectors(self, vocabulary, reqs, req_vocabulary, n):
        vectors = []
        for idx, req in enumerate(reqs):
            vector = []
            for word in vocabulary:
                if word[0] in req:
                    tf = req_vocabulary[idx][word[0]]
                    idf = math.log(n / word[1], 2)
                    vector.append(tf * idf)
                else:
                    vector.append(0)
            vectors.append(vector)
        return vectors

    def create_matrix(self, low_reqs, high_reqs):
        matrix = []
        for high_req in high_reqs:
            row = []
            for low_req in low_reqs:
                similarity = self.cosine_similarity(high_req, low_req)
                row.append(similarity)
            matrix.append(row)
        return matrix

    def create_pearson_matrix(self, low_reqs, high_reqs):
        matrix = []
        for high_req in high_reqs:
            row = []
            for low_req in low_reqs:
                similarity = self.pearson_similarity(high_req, low_req)
                row.append(similarity)
            matrix.append(row)
        return matrix







    def cosine_similarity(self, v1, v2):

        return numpy.sum(numpy.multiply(v1, v2)) / \
               (math.sqrt(numpy.sum(numpy.square(v1))) * math.sqrt(numpy.sum(numpy.square(v1))))



    def pearson_similarity(self, v1,v2):
        corr, _ = pearsonr(v1, v2)
        print(corr)

        return corr

    def write_output(self, matrix, treshold, low_ids, high_ids):
        with open('/output/links.csv', 'w+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
            fieldnames = ["id", "links"]
            writer.writerow(fieldnames)
            for idRow, row in enumerate(matrix):
                high_id = high_ids[idRow]
                links = ""
                if treshold == 0.67:
                    highest = 0
                    for col in row:
                        if col > highest:
                            highest = col
                    for idCol, col in enumerate(row):
                        if col > treshold * highest:
                            links += ', ' + low_ids[idCol]
                else:
                    for idCol, col in enumerate(row):
                        if col > treshold:

                            links += ', ' + low_ids[idCol]
                links = links[2:]
                output = [high_id, links]
                writer.writerow(output)


    def read_data(self, path):
        df = pd.read_csv(path)
        return df
