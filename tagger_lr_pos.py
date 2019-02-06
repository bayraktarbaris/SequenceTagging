# -*- coding: utf-8 -*-

from tagger_lr import TaggerLR
from data_lr_pos import DataReaderLR_POS, token_to_features


class TaggerLR_POS(TaggerLR):
    '''
    data specific tag(sent) method
    '''

    def __init__(self, params=None):
        '''
        only sets the appropriate class
        '''
        super().__init__(params)
        # DataReaderLR class for features
        self.data_class = DataReaderLR_POS

    def _to_empty_sent(self, sentence):
        '''
        convert token to [token, None, None]
        '''
        return [[token, None, None] for token in sentence]

    def token_to_features(self, sentence, index):
        '''
        tag a list of tokenized sentences
        '''
        return token_to_features(sentence, index)


if __name__ == '__main__':

    # data_dir = 'data_mini/pos/'
    data_dir = 'data/pos/'

    tagger = TaggerLR_POS()

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

    tagger.save('models/tagger_lr_pos.pickle')
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
    tagger = TaggerLR_POS()
    tagger.load('models/tagger_lr_pos.pickle')
    print()
    print(tagger.tag(['I', 'do', 'n\'t', 'think', 'it', 'matters']))
    print()
    print(tagger.tag_sents([
        ['I', 'do', 'n\'t', 'think', 'it', 'matters'],
        ['Gets', 'the', 'Job', 'Done']]))
