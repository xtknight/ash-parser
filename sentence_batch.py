'''
Provide a batch of sentences to the trainer

Maintains batch_size slots of sentences, each one with its own parser state
'''

from conll_utils import ParsedConllFile

class SentenceBatch(object):
    def __init__(self, input_corpus, batch_size=50):
        assert type(input_corpus) is ParsedConllFile
        assert len(input_corpus.sentences) > 0, 'please call read() on input_corpus beforehand'
        self.input_corpus = input_corpus
        self.batch_size = batch_size
        self.rewind()

    def rewind(self):
        # so that sentence can advance to 0 from the beginning!
        self.highest_sentence_index = -1
        self.sentences = [None for i in range(self.batch_size)]
        self.num_active = 0

    '''
    Return current number of non-null sentences in the batch
    '''
    def size(self):
        return self.num_active

    def sentence(self, index):
        assert index >= 0 and index < self.batch_size, 'batch index out of bounds'
        return self.sentences[index]

    def advanceSentence(self, index):
        assert index >= 0 and index < self.batch_size, 'batch index out of bounds'

        if self.sentences[index] == None:
            self.num_active += 1

        if (self.highest_sentence_index+1) >= len(self.input_corpus.sentences):
            # EOF reached
            self.num_active -= 1
            return False

        self.highest_sentence_index += 1
        self.sentences[index] = self.input_corpus.sentences[self.highest_sentence_index]
        return True
