# -*- coding: utf-8 -*-

from tagger_crf import TaggerCRF
from data_crf_pos import DataReaderCRF_POS, sent_to_features


class TaggerCRF_POS(TaggerCRF):
    '''
    data specific tag_sents(sents) method
    '''

    def __init__(self, params=None):
        '''
        '''
        super().__init__(params)
        # DataReaderCRF class for features
        self.data_class = DataReaderCRF_POS

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

    # data_dir = 'data_mini/pos/'
    data_dir = 'data/pos/'

    tagger = TaggerCRF_POS()

    tagger.train(data_dir + 'en-ud-train.conllu')
    print('eval acc:', tagger.evaluate(data_dir + 'en-ud-dev.conllu'))

    precision, recall, f1, accuracy, confusion, tagged_sents = \
        tagger.test(data_dir + 'en-ud-test.conllu')

    print('test pre:', precision)
    print('test rec:', recall)
    print('test f1: ', f1)
    print('test acc:', accuracy)
    print('test con:', confusion)
    print()

    # print(tagged_sents)

    tagger.save('models/tagger_crf_pos.pickle')
    del tagger
    try:
        print(tagger)
    except Exception as e:
        print(e)
    '''
    I   PRON    PRP
    do  AUX VBP
    n't PART    RB
    think   VERB    VB
    it  PRON    PRP
    matters VERB    VBZ

    Gets    VERB    VBZ
    the DET DT
    Job NOUN    NN
    Done    ADJ JJ
    '''
    tagger = TaggerCRF_POS()
    tagger.load('models/tagger_crf_pos.pickle')
    print()
    print(tagger.tag(['I', 'do', 'n\'t', 'think', 'it', 'matters']))
    print()
    print(tagger.tag_sents([
        ['I', 'do', 'n\'t', 'think', 'it', 'matters'],
        ['Gets', 'the', 'Job', 'Done']]))
