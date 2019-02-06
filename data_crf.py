# -*- coding: utf-8 -*-

from data import DataReader


class DataReaderCRF(DataReader):
    '''
    base class for data operations in conditional random fields models
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
        just call module level function
        for your specific feature set implementation
        '''
        raise NotImplementedError("sent_to_features not implemented")

    def extract_features(self):
        '''
        '''
        for sent in self.data:
            self.features.append(self.sent_to_features(sent))
