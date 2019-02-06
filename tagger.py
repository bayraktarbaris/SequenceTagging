# -*- coding: utf-8 -*-

import logging

LOG_LEVEL = logging.INFO


class Tagger():
    '''
    base class for all taggers
    '''

    def __init__(self, params=None):
        '''
        initialize the tagger (and the vectorizer if necessary)
        params is a dict of tagger arguments that can be passed to models
        '''
        self.logger = None
        self.init_logging(LOG_LEVEL)

    def train(self, file_name):
        '''
        extract the features and train the model
        file_name shows the path to the file with task specific format
        '''
        raise NotImplementedError("train not implemented")

    def test(self, file_name, labels_to_remove=[]):
        '''
        test the model, extract features and predict tags
        return metrics, confusion matrix and tagged data in the order below
        precision
        recall
        f1
        accuracy
        confusion matrix
        tagged data
        file_name shows the path to the file with task specific format
        labels_to_remove show classes to ignore
            while calculating precision, recall and f1 scores
        '''
        raise NotImplementedError("test not implemented")

    def evaluate(self, file_name, labels_to_remove=[]):
        '''
        return accuracy
        to be used for validation, learning curve, parameter optimization etc.
        '''
        return self.test(file_name, labels_to_remove)[3]

    def tag(self, sentence):
        '''
        tag a single tokenized sentence
        sentence: [tkn1, tkn2, ... tknN]
        return the tagged sentence as list of fields
            as given in training/test files
        tagged sentence: [[tkn1, ..., tag1], [tkn2, ..., tag2], ...]
        put None for any field between token and tag that is not predicted
        '''
        raise NotImplementedError("tag not implemented")

    def tag_sents(self, sentences):
        '''
        tag a list of tokenized sentences
        sentences: [[tkn11, tkn12, ... tkn1N], [tkn21, tkn22, ... tkn2M] ...]
        return the tagged sentence as a list of fields
            as given in training/test files
        tagged sentences: [[[tkn11, ..., tag11], ...], ...]
        put None for any field between token and tag that is not predicted
        '''
        raise NotImplementedError("tag_sents not implemented")

    def save(self, file_name):
        '''
        save the trained models for tagger to file_name
        '''
        raise NotImplementedError("save not implemented")

    def load(self, file_name):
        '''
        load the trained models for tagger from file_name
        '''
        raise NotImplementedError("load not implemented")

    def init_logging(self, log_level):
        '''
        logging config and init
        '''
        if not self.logger:
            logging.basicConfig(
                format='%(asctime)s-|%(name)20s:%(funcName)12s|'
                       '-%(levelname)8s-> %(message)s')
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(log_level)
