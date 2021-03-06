# -*- coding: utf-8 -*-

from tagger_crf import TaggerCRF
from data_crf_ner import DataReaderCRF_NER, sent_to_features


class TaggerCRF_NER(TaggerCRF):
    '''
    data specific tag_sents(sents) method
    '''

    def __init__(self, params=None):
        '''
        '''
        super().__init__(params)
        # DataReaderCRF class for features
        self.data_class = DataReaderCRF_NER

    def _to_empty_sent(self, sentence):
        '''
        convert token to [token, None, None]
        '''
        return [[token, None, None, None] for token in sentence]

    def sent_to_features(self, sentence):
        '''
        tag a list of tokenized sentences
        '''
        return sent_to_features(sentence)

    def test(self, file_name, labels_to_remove=None):
        '''
        precision, recall and f1 scores without O label
        '''
        return super().test(file_name, labels_to_remove=['O'])


if __name__ == '__main__':

    # data_dir = 'data_mini/ner/'
    data_dir = 'data/ner/'

    tagger = TaggerCRF_NER()

    tagger.train(data_dir + 'eng.train.txt')
    print('eval acc:', tagger.evaluate(data_dir + 'eng.testa.txt'))

    precision, recall, f1, accuracy, confusion, tagged_sents = \
        tagger.test(data_dir + 'eng.testb.txt')

    print('test pre:', precision)
    print('test rec:', recall)
    print('test f1: ', f1)
    print('test acc:', accuracy)
    print('test con:', confusion)
    print()

    # print(tagged_sents)

    tagger.save('models/tagger_crf_ner.pickle')
    del tagger
    try:
        print(tagger)
    except Exception as e:
        print(e)
    '''
    --    :    B-NP    O
    Brussels    NNP    I-NP    B-ORG
    Newsroom    NNP    I-NP    I-ORG
    32    CD    I-NP    O
    2    CD    I-NP    O
    287    CD    I-NP    O
    6800    CD    I-NP    O

    There    EX    B-NP    O
    was    VBD    B-VP    O
    no    DT    B-NP    O
    Bundesbank    NNP    I-NP    B-ORG
    intervention    NN    I-NP    O
    .    .    O    O
    '''
    tagger = TaggerCRF_NER()
    tagger.load('models/tagger_crf_ner.pickle')
    print()
    print(tagger.tag(['--', 'Brussels', 'Newsroom', '32', '2', '287', '6800']))
    print()
    print(tagger.tag_sents([
        ['--', 'Brussels', 'Newsroom', '32', '2', '287', '6800'],
        ['There', 'was', 'no', 'Bundesbank', 'intervention', '.']]))
