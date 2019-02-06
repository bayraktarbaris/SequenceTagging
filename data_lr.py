# -*- coding: utf-8 -*-

from data import DataReader


class DataReaderLR(DataReader):
    '''
    base class for data operations in logistic regression models
    '''

    def __init__(self, file_name):
        '''
        '''
        super().__init__(file_name)

    def token_to_features(self, sentence):
        '''
        just call module level function
        for your specific feature set implementation
        '''
        raise NotImplementedError("token_to_features not implemented")

    def sent_to_features(self, sentence):
        '''
        tokenized sentence
        '''
        sent_feats = []
        for i, _ in enumerate(sentence):
            sent_feats.append(self.token_to_features(sentence, i))
        return sent_feats

    def extract_features(self):
        '''
        '''
        for sent in self.data:
            self.features.extend(self.sent_to_features(sent))

    def extract_labels(self):
        '''
        '''
        for sent in self.data:
            self.labels.extend([fields[-1] for fields in sent])
