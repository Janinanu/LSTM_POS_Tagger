#! /usr/bin/python3

import argparse
from Data import Data
import torch

arg_parser = argparse.ArgumentParser(description='LSTMTagger annotation module.\n'
                                                 'Loads model files and annotates sentence file (one word per line)\n'
                                                 'Prints the annotations to stdout (one tag per line)')
arg_parser.add_argument('path_param', type=str, help='path containing the model files (.io and .rnn)')
arg_parser.add_argument('test_file', type=str, help='a file containing the data to be annotated')
arg_parser.add_argument('--gpu', action='store_true',
                        help='if this parameter is present, computation will be performed on gpu, otherwise on cpu')
args = arg_parser.parse_args()

# Load the symbol mapping tables
data = Data( args.path_param+".io", args.gpu )
# Load the model
model = torch.load( args.path_param+".rnn" )

# use model on gpu or cpu, regardless of the training settings
model.cuda() if args.gpu else model.cpu()

# annotate sentences
for sentence, _ in data.sentences(args.test_file):
    scores = model(data.words2IDs(sentence))
    _, predictions = scores.max(dim=-1) # returns Variable
    predicted_tags = data.IDs2tags(predictions) # returns list
    for word, tag in zip(sentence, predicted_tags):
        print(word, tag, sep="\t")
    print()
