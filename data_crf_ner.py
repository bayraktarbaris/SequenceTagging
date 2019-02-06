# -*- coding: utf-8 -*-

from data_crf import DataReaderCRF


def token_to_features(sentence, i):
    '''
    '''
    token = sentence[i][0]
    token_feats = {'bias': 1.0}
    token_feats['word'] = token.lower()

    return token_feats


def sent_to_features(sentence):
    '''
    tokenized sentence
    '''
    sent_feats = []
    for i, _ in enumerate(sentence):
        sent_feats.append(token_to_features(sentence, i))
    return sent_feats


class DataReaderCRF_NER(DataReaderCRF):
    '''
    read the contents of the files
    extract features
    return labels etc.
    '''

    def __init__(self, file_name):
        '''
        '''
        super().__init__(file_name)
        # list of list of tokens
        self.sentences = []
        # list of token features
        self.features = []
        # list of token tags
        self.labels = []
        self.extract()
        # alternative names
        self.X = self.features
        self.y = self.labels

    def token_to_features(self, sentence, i):
        return token_to_features(sentence, i)

    def sent_to_features(self, sentence):
        return sent_to_features(sentence)


if __name__ == '__main__':
    data = DataReaderCRF_NER('data_mini/ner/eng.train.txt')
    print(len(data.data))
    print()
    print(data.data[0])
    print()
    print(data.sentences[0])
    print()
    print(data.features[0])
    print()
    print(data.labels[0])
    print()
    print(data.X[0])
    print()
    print(data.y[0])
    print()
    print(len(data.X))
    print()
    print(len(data.y))
    print()
