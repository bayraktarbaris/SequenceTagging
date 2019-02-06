# -*- coding: utf-8 -*-

from tagger_crf import TaggerCRF
from data_crf_chunk import DataReaderCRF_Chunk, sent_to_features


class TaggerCRF_Chunk(TaggerCRF):
    '''
    data specific tag_sents(sents) method
    '''

    def __init__(self, params=None):
        '''
        '''
        super().__init__(params)
        # DataReaderCRF class for features
        self.data_class = DataReaderCRF_Chunk

    def _to_empty_sent(self, sentence):
        '''
        convert token to [token, None, None]
        '''
        return [[token, None, None] for token in sentence]

    def sent_to_features(self, sentence):
        '''
        tag a list of tokenized sentences
        '''
        return sent_to_features(sentence)


if __name__ == '__main__':

    # data_dir = 'data_mini/chunk/'
    data_dir = 'data/chunk/'

    tagger = TaggerCRF_Chunk()

    tagger.train(data_dir + 'train.txt')
    print('eval acc:', tagger.evaluate(data_dir + 'train.txt'))

    precision, recall, f1, accuracy, confusion, tagged_sents = \
        tagger.test(data_dir + 'test.txt')

    print('test pre:', precision)
    print('test rec:', recall)
    print('test f1: ', f1)
    print('test acc:', accuracy)
    print('test con:', confusion)
    print()

    # print(tagged_sents)

    tagger.save('models/tagger_crf_chunk.pickle')
    del tagger
    try:
        print(tagger)
    except Exception as e:
        print(e)
    '''
    Mr.    NNP    B-NP
    Noriega    NNP    I-NP
    was    VBD    B-VP
    growing    VBG    I-VP
    desperate    JJ    B-ADJP
    .    .    O

    The    DT    B-NP
    end    NN    I-NP
    of    IN    B-PP
    the    DT    B-NP
    marriage    NN    I-NP
    was    VBD    B-VP
    at    IN    B-PP
    hand    NN    B-NP
    .    .    O
    '''
    tagger = TaggerCRF_Chunk()
    tagger.load('models/tagger_crf_chunk.pickle')
    print()
    print(tagger.tag(['Mr.', 'Noriega', 'was', 'growing', 'desperate', '.']))
    print()
    print(tagger.tag_sents([
        ['Mr.', 'Noriega', 'was', 'growing', 'desperate', '.'],
        ['The', 'end', 'of', 'the', 'marriage', 'was', 'at', 'hand', '.']]))
