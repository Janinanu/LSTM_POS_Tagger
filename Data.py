#! /usr/bin/python3

from collections import Counter
import argparse
import pickle
import torch
from torch import autograd


class Data:
    """
    Class that processes and represents the training and development data
    as well as creates and applies the dictionaries required to map words and tags to IDs
    """

    def __init__(self, *args):
        if len(args) == 2:
            self._init_test(*args)
        else:
            self._init_train(*args)

    def _init_train(self, traindata, devdata, numWords, use_gpu):
        """ Constructor for training"""
        self.gpu = use_gpu
        self.trainSentences = [(words, tags) for words, tags in self.sentences(traindata)]
        self.devSentences = [(words, tags) for words, tags in self.sentences(devdata)]
        self.num_tags = 0
        self.word_to_id = {}
        self.tag_to_id = {}
        self.id_to_tag = []
        self._create_dicts(numWords)

    def _init_test(self, parameter_file, use_gpu):
        """ Constructor for testing
        :param parameter_file: file that contains the stored dictionaries to map from strings to IDs
        """
        self.gpu = use_gpu
        with open(parameter_file, 'rb') as file:
            self.word_to_id = pickle.load(file)
            self.id_to_tag = pickle.load(file)
        self.tag_to_id = {tag: id for id, tag in enumerate(self.id_to_tag)}

    def _create_dicts(self, numWords):
        """Processes training data and creates the dictionaries word_to_id, tag_to_id, and id_to_tag
        :param numWords: the number of most frequent words to take into account
        """
        word_freq = Counter()
        distinct_tags = set() #to represent classes
        for words, tags in self.trainSentences:
            distinct_tags.update(tags)
            word_freq.update(words)

        most_freq_words = [word for word, _ in word_freq.most_common(numWords)]
        # start at ID 1 to reserve 0 for words not represented in the numWords most frequent words
        self.word_to_id = {word: id for id, word in enumerate(most_freq_words, 1)}
        # start at ID 1 to reserve 0 for tags not seen during training
        self.tag_to_id = {tag: id for id, tag in enumerate(distinct_tags, 1)}
        # add <UNK> class at ID 0 to map to tags not seen during training
        self.id_to_tag =  ["<UNK>"] + list(distinct_tags)
        self.numTags = len(self.id_to_tag) #number of all classes including unknown class

    def _to_variable(self, list):
        """
        Turns a list of word IDs or tag IDs into PyTorch-specific LongTensor Variable, optionally GPU processable
        :param IDs: a list of IDs representing the words or tags in a sentence
        :return: tensor Variable of IDs, of dimension len(IDs)
        """
        variable = autograd.Variable(torch.LongTensor(list))
        return variable.cuda() if self.gpu else variable

    def to_array(self, variable):
        """
        Extracts data from a PyTorch-specific LongTensor Variable
        :param variable: Variable of class autograd
        :return: array containing the data from the variable
        """
        return variable.data.cpu() if self.gpu else variable.data

    def words2IDs(self, words):
        """
        Maps each word in a sentence to its ID, maps unknown words to 0
        :param words: a list of strings representing a sentence
        :return: a tensor Variable of word IDs
        """
        return self._to_variable([self.word_to_id.get(word, 0) for word in words])

    def tags2IDs(self, tags):
        """
        Maps each tag in a sentence to its ID, maps unknown tags to 0
        :param tags: a list of strings representing the tags in a sentence
        :return: a tensor Variable of tag IDs
        """
        return self._to_variable([self.tag_to_id.get(tag, 0) for tag in tags])

    def IDs2tags(self, ids):
        """
        Maps IDs to their corresponding tags, maps ID 0 to "<UNK>" class
        :param ids: a LongTensor Variable of IDs
        :return: a list of tags
        """
        return [self.id_to_tag[id] for id in self._to_array(ids)]

    def store_parameters(self, file):
        """
        Save the the symbol mapping tables to a binary file
        :param file: the file in which we store the dictionaries
        """
        with open(file, "wb") as parameter_file:
            pickle.dump(self.word_to_id, parameter_file)
            pickle.dump(self.id_to_tag, parameter_file)

    @staticmethod
    def sentences(sentence_file):
        """
        Generator that reads in a file and yields list of words and tags (if present) after each sentence
        :param sentence_file: a file containing the test sentences
        :return: yields each sentence and its tags (if present, else the tag list is empty)
        """
        words = []
        tags = []
        with open(sentence_file) as f:
            for line in f:
                if line != "\n":
                    if "\t" in line:
                        word, tag = line.strip().split("\t")
                        words.append(word)
                        tags.append(tag)
                    else:
                        words.append(line.strip())
                else:
                    yield words, tags
                    words = []
                    tags = []

def run_test():
    """
    Creates data object and tests its functionality,
    prints out words and tags with their respective IDs for the first 5 training sentences
    """
    arg_parser = argparse.ArgumentParser(description='Class for reading in data')
    arg_parser.add_argument('traindata', type=str, help='a file containing the training data')
    arg_parser.add_argument('devdata', type=str, help='a file containing the evaluation data')
    arg_parser.add_argument('numWords', type=int, help='the number of most frequent words to take into account')
    arg_parser.add_argument('--gpu', action='store_true',
                            help='if this parameter is present, computation will be performed on gpu, otherwise on cpu')
    args = arg_parser.parse_args()

    data = Data(args.traindata, args.devdata, args.numWords, args.gpu)

    for words, tags in data.trainSentences[:5]: #limit test print out to first 5 sentences
        wordIDs = data.words2IDs(words)
        tagIDs = data.tags2IDs(tags)
        assert len(wordIDs) == len(tagIDs)
        for word, wordID, tag, tagID in zip(words, wordIDs, tags, tagIDs):
            print(word, wordID.data[0], tag, tagID.data[0])

if __name__ == "__main__":
    run_test()
