# -*- coding: utf-8 -*-

import logging

LOG_LEVEL = logging.INFO


class DataReader():
    '''
    base class for all data operations
    '''

    def __init__(self, file_name):
        '''
        '''
        self.file_name = file_name
        # list of list of token, feature, label tuples
        # (as fields in the original data)
        self.data = []
        # list of list of tokens
        self.sentences = []
        # list of token features
        self.features = []
        # list of token tags
        self.labels = []
        #
        self.logger = None
        self.init_logging(LOG_LEVEL)
        self.logger.info('processing ' + self.file_name)
        self.read_data()

    def read_data(self):
        '''
        '''
        with open(self.file_name, 'r') as in_file:
            sent = []
            for line in in_file:
                if line.strip() == '':
                    if sent == []:
                        break
                    self.data.append(sent)
                    sent = []
                else:
                    sent.append(line.strip().split('\t'))
        self.logger.info('read data')

    def extract(self):
        '''
        convert data
        '''
        self.extract_sentences()
        self.extract_features()
        self.extract_labels()
        self.logger.info('extracted features')

    def extract_sentences(self):
        '''
        '''
        self.sentences = [[fields[0] for fields in sent] for sent in self.data]

    def extract_features(self):
        '''
        '''
        raise NotImplementedError("extract_features not implemented")

    def extract_labels(self):
        '''
        '''
        self.labels = [[fields[-1] for fields in sent] for sent in self.data]

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
