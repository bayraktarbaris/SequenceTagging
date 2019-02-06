# -*- coding: utf-8 -*-

from data_lr import DataReaderLR


def token_to_features(sentence, i):
    '''
    module level function that can be called without an object
    this is the feature extractor
    data specific
    '''
    token_feats = {}
    token_feats['word'] = sentence[i][0].lower()
    if i > 0:
        token_feats['pos_p'] = sentence[i - 1][2]
    else:
        token_feats['pos_p'] = 'BGN'
    return token_feats


class DataReaderLR_Chunk(DataReaderLR):
    '''
    read the contents of the files
    extract features
    return tokens, feature, labels etc.
    '''

    def __init__(self, file_name):
        '''
        '''
        super().__init__(file_name)
        self.extract()
        # alternative names
        self.X = self.features
        self.y = self.labels

    def token_to_features(self, sentence, i):
        return token_to_features(sentence, i)


if __name__ == '__main__':
    data = DataReaderLR_Chunk('data_mini/chunk/train.txt')
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
