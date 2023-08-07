# Script to read options
# author: satwik kottur
import argparse
import pdb
import os

# read command line arguments
def read():
    parser = argparse.ArgumentParser(description='RL Train toy example')

    # Model parameters
    parser.add_argument('-ix', default=0, type=int,\
                            help='index for saving path')
    parser.add_argument('-cHiddenSize', default=50, type=int,\
                            help='Hidden size for child\'s language model')
    parser.add_argument('-mHiddenSize', default=50, type=int,\
                            help='Hidden size for mother\'s language model')
    parser.add_argument('-embedSize', default=20, type=int,\
                            help='Embed size for words')
    parser.add_argument('-imgFeatSize', default=20, type=int,\
                            help='Image feature size for each attribute')
    parser.add_argument('-cOutVocab', default=3, type=int,\
                            help='Output vocabulary for child')
    parser.add_argument('-mOutVocab', default=4, type=int,\
                            help='Output vocabulary for mother')
    #parser.add_argument('-mInsightSize', default=0, type=int,\
    #                        help='Insight size for mother')
    parser.add_argument('-cSpeakFreq', default=1, type=int,\
                            help='Frequency of child messages')
    parser.add_argument('-mActFreq', default=1, type=int,\
                            help='Frequency of mother actions')
    parser.add_argument('-numVitalParams', default=5, type=int,\
                            help='Number of childs vital parameters')
    parser.add_argument('-decay', default='log', type=str,\
                            help='Type of decay of childs vital parameters')
    #parser.add_argument('-dataset', default='data/64_synthetic.json',\
    #                        type=str, help='Path to the dataset')
    parser.add_argument('-rlScale', default=100.0, type=float,\
                            help='Weight given to rl gradients')
    parser.add_argument('-numRounds', default=2, type=int,\
                            help='Number of rounds between Q and A')
    parser.add_argument('-remember', dest='remember', action='store_true', \
                            help='Turn on/off for ABot with memory')
    #parser.add_argument('-negFraction', default=0.8, type=float,\
    #                        help='Fraction of negative examples in batch')

    # Optimization options
    parser.add_argument('-batchSize', default=1000, type=int,\
                            help='Batch size -- number of episodes')
    parser.add_argument('-numEpochs', default=1000000, type=int,\
                            help='Maximum number of epochs to run')
    parser.add_argument('-learningRate', default=5e-4, type=float,\
                            help='Initial learning rate')
    parser.add_argument('-useGPU', dest='useGPU', action='store_true')

    try:
      parsed = vars(parser.parse_args())
    except IOError as err:
      parser.error(str(err))

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in parsed.items():
      print(fmtString % keyPair)

    return parsed
