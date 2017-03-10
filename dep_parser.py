from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pickle
import copy
import logging
import math
import tensorflow as tf
import numpy as np
import sys
import os
import argparse

from model_parameters import *
from lexicon import *
from conll_utils import *
from feature_extractor import SparseFeatureExtractor
from parser_state import ParserState
from arc_standard_transition_system import ArcStandardTransitionState, \
     ArcStandardTransitionSystem
from gold_parse_reader import GoldParseReader
from decoded_parse_reader import DecodedParseReader

logger = logging.getLogger('DepParser')

parser = argparse.ArgumentParser(
    description='Train a Chen and Manning-style neural network dependency' \
                'parser')

# Required positional argument
parser.add_argument('model_folder', type=str,
                    help='Folder in which to load or save model')
parser.add_argument('training_file', type=str,
                    help='CoNLL-U format tagged training corpus (UTF-8)')
parser.add_argument('testing_file', type=str,
                    help='CoNLL-U format tagged evaluation corpus (UTF-8)')
parser.add_argument('--train', action='store_true', default=False,
                    help='Training a new model or continue training of an '
                         'old model')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='Evaluate an existing model')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Enable verbose debug lines')
#parser.add_argument('--epochs', type=int, default=10,
#                    help='Training epochs (default 10). Shuffle sentences '
#                         ' and re-train during each training epoch.')

## TODO:
# add param: use pretrained word/sense embeddings gensim/Mikolov
# projectivize

args = parser.parse_args()

if args.debug:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
else:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

try:
    os.makedirs(args.model_folder)
except:
    pass

if (args.train and args.evaluate) or ((not args.train) and (not args.evaluate)):
    print('Please specify either training or evaluation mode '
          '(--train/--evaluate)')
    sys.exit(1)

'''
Entry point for dependency parser
'''
class Parser(object):
    def __init__(self, modelParams, mode):
        self.logger = logging.getLogger('Parser')
        self.modelParams = modelParams
        self.setupParser(mode)
        self.setupVariables(mode)
    
        self.trainingCorpus = None
        self.testingCorpus = None

    '''
    Setup transition and action system and feature maps
    (necessary whether training or evaluating)
    '''
    def setupParser(self, mode='train'):
        hiddenLayerSizes = self.modelParams.cfg['hiddenLayerSizes']
        featureStrings = self.modelParams.cfg['featureStrings']
        embeddingSizes = self.modelParams.cfg['embeddingSizes']
        batchSize = self.modelParams.cfg['batchSize']

        self.transitionSystem = ArcStandardTransitionSystem()

        assert len(hiddenLayerSizes) > 0, 'must have at least one hidden layer'
        assert len(featureStrings) == len(set(featureStrings)), \
            'duplicate feature string detected'

        if mode == 'train':
            # determine if we have to compute or read the lexicon
            self.logger.info('Computing lexicon from training corpus...')
            self.modelParams.lexicon.compute()
            self.logger.info('Done building lexicon')
            self.modelParams.lexicon.write()
        elif mode == 'evaluate':
            self.logger.info('Reading lexicon from trained model...')
            self.modelParams.lexicon.read()
        else:
            assert None, 'invalid mode: ' + mode

        self.featureMaps = self.modelParams.lexicon.getFeatureMaps()

        self.logger.info('Feature strings: ' + str(featureStrings))

        # Get major type groups in sorted order by contructing null parser
        # state and extracting features, and then concatenating the similar
        # types
        fvec = SparseFeatureExtractor(featureStrings, self.featureMaps) \
            .extract(ParserState(ConllSentence(), self.featureMaps))

        featureTypeInstances = fvec.types
        self.featureMajorTypeGroups, _ = fvec.concatenateSimilarTypes()

        # index: major feature type index
        # values: feature names under that type
        self.featureNames = [[] for t in self.featureMajorTypeGroups]

        self.logger.info('Detected major feature groups (in alphabetical '
            'order): ' + str(self.featureMajorTypeGroups))

        self.featureDomainSizes = []
        self.featureEmbeddings = []

        # For now, use all same embedding sizes
        self.featureEmbeddingSizes = \
            [embeddingSizes[t] for t in self.featureMajorTypeGroups] 

        self.BAG_OF_FEATURES_LEN = 0

        for i in range(len(featureTypeInstances)):
            major_type = featureTypeInstances[i].major_type
            major_type_index = self.featureMajorTypeGroups.index(major_type)

            self.featureNames[major_type_index].append(
                featureTypeInstances[i].name)

            self.BAG_OF_FEATURES_LEN += \
                (self.featureEmbeddingSizes[major_type_index])

        for i in range(len(self.featureMajorTypeGroups)):
            major_type = self.featureMajorTypeGroups[i]

            self.logger.info('')
            self.logger.info('Feature group \'%s\'' % major_type)
            self.logger.info('... domain size: %d' % \
                (self.featureMaps[major_type].getDomainSize()))
            self.logger.info('... embedding size: %d' % \
                (self.featureEmbeddingSizes[i]))
            #self.logger.info('... feature count: %d' % \
            #    (len(self.featureNames[i])))
            self.logger.info('... features')

            for fname in self.featureNames[i]:
                self.logger.info('....... %s' % (fname))

            self.logger.info('... total group embedding size: %d' % \
                (len(self.featureNames[i]) * self.featureEmbeddingSizes[i]))

            self.logger.info('... initializing random normal embeddings...')
            self.featureDomainSizes.append(
                self.featureMaps[major_type].getDomainSize())

            with tf.variable_scope("feature_%s" % major_type):
                self.featureEmbeddings.append(tf.get_variable("embeddings", \
                    [self.featureMaps[major_type].getDomainSize(), \
                    self.featureEmbeddingSizes[i]], \
                    initializer=tf.random_normal_initializer(
                        stddev=1.0 / self.featureEmbeddingSizes[i]**.5, \
                        seed=0)))

        assert len(self.featureDomainSizes) == len(self.featureEmbeddingSizes)
        assert len(self.featureDomainSizes) == len(self.featureEmbeddings)
        assert len(self.featureDomainSizes) == len(self.featureNames)

        self.logger.info('')
        self.logger.info('Batch size (number of parser states): %d' % batchSize)
        self.logger.info('Total feature count: %d' % \
            (len(featureTypeInstances)))
        self.logger.info('Total bag of features length per state: %d' % \
            (self.BAG_OF_FEATURES_LEN))
        self.logger.info('Total features input size: %d' % \
            (batchSize*self.BAG_OF_FEATURES_LEN))

        # we'll use parser state's version of the action encoding
        self.ACTION_COUNT = self.transitionSystem.numActions(
            self.featureMaps['label'].getDomainSize())

        self.ACTION_ARRAY = [n for n in range(self.ACTION_COUNT)]

        self.logger.info('Total action count: %d' % self.ACTION_COUNT)

    '''
    Setup TensorFlow Variables in model
    '''
    def setupVariables(self, mode='train'):
        learningRate = self.modelParams.cfg['learningRate']
        topK = self.modelParams.cfg['topK']
        hiddenLayerSizes = self.modelParams.cfg['hiddenLayerSizes']

        # Bag of features at each parser state
        self.X = tf.placeholder(tf.float32, [None, self.BAG_OF_FEATURES_LEN], \
            name="ph_X")

        # Next action to take for given input parser state
        # (One-hot encoding)
        # SHIFT, LEFT_ARC(L), or RIGHT_ARC(L)
        # Output: (, 3+~50)
        self.Y = tf.placeholder(tf.float32, [None, self.ACTION_COUNT], \
            name="ph_Y")

        self.weights = []
        self.biases = []

        # Input layer
        prevLayerSize = self.BAG_OF_FEATURES_LEN

        # Hidden layers
        for i in range(len(hiddenLayerSizes)):
            h = hiddenLayerSizes[i]

            with tf.variable_scope("layer_%d" % i):
                self.weights.append(tf.get_variable("weights", \
                    [prevLayerSize, h], \
                    initializer=tf.random_normal_initializer(
                        stddev=1e-4, seed=0)))

                self.biases.append(tf.get_variable("biases", [h], \
                    initializer=tf.constant_initializer(0.2)))

            prevLayerSize = h
        
        # Output layer
        with tf.variable_scope("layer_out"):
            self.weights.append(tf.get_variable("weights", \
                [prevLayerSize, self.ACTION_COUNT], \
                initializer=tf.random_normal_initializer(stddev=1e-4, seed=0)))

            self.biases.append(tf.get_variable("biases", \
                [self.ACTION_COUNT], initializer=tf.constant_initializer(0.2)))
        
        def greedy_graph_builder(x, weights, biases):
            hidden_layers = []
            last_layer = x

            # Input and hidden layers
            for i in range(len(hiddenLayerSizes)):
                this_layer = \
                    tf.add(tf.matmul(last_layer, weights[i]), biases[i])

                this_layer = tf.nn.relu(this_layer)

                # FIXME: decide whether or not to do drop-out
                #this_layer = tf.nn.dropout(this_layer, 0.5)

                hidden_layers.append(this_layer)
                last_layer = hidden_layers[-1]

            # Output layer
            out_layer = tf.add(tf.matmul(last_layer, weights[-1]), biases[-1])
            return hidden_layers, out_layer

        # Construct model
        self.hidden_layers, self.out_layer = \
            greedy_graph_builder(self.X, self.weights, self.biases)

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.out_layer, self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learningRate).minimize(self.cost)
        self.pred = tf.nn.softmax(self.out_layer)
        self.pred_top_k = tf.nn.top_k(self.pred, k=topK)


    '''
    Start training where we left off
    '''
    def startTraining(self):
        batchSize = self.modelParams.cfg['batchSize']
        featureStrings = self.modelParams.cfg['featureStrings']

        self.trainingCorpus = ParsedConllFile()
        self.trainingCorpus.read(open(self.modelParams.trainingFile, 'r',
                          encoding='utf-8').read())

        # Start getting sentence batches...
        reader = GoldParseReader(self.trainingCorpus, batchSize, \
            featureStrings, self.featureMaps)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        #config.gpu_options.per_process_gpu_memory_fraction=1.0

        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()

        epoch_num_old = 0
        avg_cost = 0.0

        ckpt_dir = fixPath(self.modelParams.modelFolder) + '/'

        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            sess.run(init_op)

            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restore variables from disk.
                saver.restore(sess, ckpt.model_checkpoint_path)
                self.logger.info('Model restored')
                self.logger.info('Continue fitting')

            # every 50 batches, reset avg and print
            print_freq = 50
            save_freq = 50
            eval_freq = 200

            i = 0
            while(True):
                self.logger.debug('Iter(%d): nextFeatureBags()' % i)
                reader_output = reader.nextFeatureBags()
                if reader_output[0] == None:
                    self.logger.debug('Iter(%d): reader output is None' % i)
                    break

                '''
                    epoch_num refers to the number of run-throughs through the
                    whole training corpus, whereas `i` is just the batch
                    iteration number
                '''
                features_major_types, features_output, gold_actions, \
                    epoch_num = reader_output

                embeddings = []

                # order of features will match the order of our arrays above
                for k in range(len(features_output)):
                    embeddings.append(tf.nn.embedding_lookup(
                        self.featureEmbeddings[k], features_output[k]))

                    embeddings[-1] = tf.reshape(embeddings[-1], \
                        [-1, len(self.featureNames[k]) * \
                        self.featureEmbeddingSizes[k]])
                
                tf_batch_x = tf.concat(1, embeddings)

                tf_batch_y = tf.one_hot(indices=tf.constant(gold_actions), \
                                        depth=self.ACTION_COUNT, on_value=1.0, \
                                        off_value=0.0, axis=-1)

                _, c = sess.run([self.optimizer, self.cost], \
                                feed_dict={self.X: tf_batch_x.eval(),
                                self.Y: tf_batch_y.eval()})

                # divide by number of actual batch items returned
                avg_cost += c / len(gold_actions)

                if i % print_freq == 0:
                    self.logger.info('Epoch: %04d Iter: %06d cost=%s' % \
                        (epoch_num, i+1, "{:.9e}".format(avg_cost)))
                    self.doEvaluation(sess)
                    # reset avg
                    avg_cost = 0.0

                if i > 0 and i % save_freq == 0:
                    save_path = saver.save(sess, ckpt_dir + 'model.ckpt')
                    self.logger.info('Model saved to file: %s' % save_path)

                if i > 0 and i % eval_freq == 0:
                    self.doEvaluationAndPrintOutput(sess)

                epoch_num_old = epoch_num

                i += 1

    '''
    Evaluate a trained model

    FIXME: invalid because we initialize parser labels with GoldParseReader
    '''
    def doEvaluation(self, sess):
        batchSize = self.modelParams.cfg['batchSize']
        featureStrings = self.modelParams.cfg['featureStrings']

        #if not self.testingCorpus:
        self.testingCorpus = ParsedConllFile()
        self.testingCorpus.read(open(self.modelParams.testingFile, 'r',
                                     encoding='utf-8').read())

        # Start getting sentence batches...
        test_reader = GoldParseReader(self.testingCorpus, batchSize, \
            featureStrings, self.featureMaps, epoch_print=False)

        correctActions = 0
        correctElems = 0
        totalElems = 0

        test_runs = 10
        for i in range(test_runs):
            test_reader_output = test_reader.nextFeatureBags()
            if test_reader_output[0] == None:
                logger.critical('Reader error')
                return

            features_major_types, features_output, gold_actions, \
                epoch_num = test_reader_output

            embeddings = []

            # order of features will match the order of our arrays above
            for k in range(len(features_output)):
                embeddings.append(tf.nn.embedding_lookup(
                    self.featureEmbeddings[k], features_output[k]))

                embeddings[-1] = tf.reshape(embeddings[-1], \
                    [-1, len(self.featureNames[k]) * \
                    self.featureEmbeddingSizes[k]])
            
            tf_batch_x = tf.concat(1, embeddings)

            Y_actual = tf.one_hot(indices=tf.constant(gold_actions), \
                depth=self.ACTION_COUNT, on_value=1.0, off_value=0.0, \
                axis=-1).eval()

            Y_pred = sess.run(self.pred, feed_dict={self.X: tf_batch_x.eval()})

            for i in range(len(Y_actual)):
                # decode one-hot

                predActionCombined = np.argmax(Y_pred[i])
                actualActionCombined = np.argmax(Y_actual[i])

                predAction = self.transitionSystem.actionType(
                    predActionCombined)

                actualAction = self.transitionSystem.actionType(
                    actualActionCombined)

                if predAction == ParserState.SHIFT:
                    predTag = None
                else:
                    predTag = self.transitionSystem.label(predActionCombined)

                if actualAction == ParserState.SHIFT:
                    actualTag = None
                else:
                    actualTag = self.transitionSystem.label(
                        actualActionCombined)

                if predAction == actualAction:
                    correctActions += 1

                if predAction == actualAction and predTag == actualTag:
                    correctElems += 1

                totalElems += 1

        # also print out number of correct actions (even if tag was wrong)
        # errors that don't accumulate (tokens are tested individually with
        # gold inputs)
        self.logger.info('Non-Propagated Error Measure (test_set)')
        self.logger.info('Actions+Labels: %d/%d (%.2f%%), '
                         'Actions: %d/%d (%.2f%%)' % \
                         (correctElems, totalElems, 100.0 * \
                         (float(correctElems) / float(totalElems)), \
                         correctActions, totalElems, 100.0 * \
                         (float(correctActions) / float(totalElems))))

    def beamSearchExperiment(self, sess):
        batchSize = self.modelParams.cfg['batchSize']
        featureStrings = self.modelParams.cfg['featureStrings']
        topK = self.modelParams.cfg['topK']

        #if not self.testingCorpus:
        self.testingCorpus = ParsedConllFile()
        self.testingCorpus.read(open(self.modelParams.testingFile, 'r',
                                         encoding='utf-8').read())

        # evaluate sentence-wide accuracy by UAS and LAS
        # of course, token errors can accumulate and this is why sentence-wide
        # accuracy is lower
        # than token-only accuracy given by doEvaluation()

        # batch size set at one temporarily to get one sentence
        test_reader_decoded = DecodedParseReader(self.testingCorpus, 1, \
            featureStrings, self.featureMaps, epoch_print=False)

        correctActions = 0
        correctElems = 0
        totalElems = 0

        outputs = []

        filled_count = 0
        pred_top_k_var = []
        
        test_runs = 200
        for i in range(test_runs):
            test_reader_output = test_reader_decoded.nextFeatureBags(
                pred_top_k_var, filled_count)

            if test_reader_output[0] == None:
                print('eval error')
                return
            
            features_major_types, features_output, epochs, \
                filled_count = test_reader_output

            embeddings = []

            # order of features will match the order of our arrays above
            for k in range(len(features_output)):
                embeddings.append(tf.nn.embedding_lookup(
                    self.featureEmbeddings[k], features_output[k]))

                embeddings[-1] = tf.reshape(embeddings[-1], \
                    [-1, len(self.featureNames[k]) * \
                    self.featureEmbeddingSizes[k]])
            
            tf_batch_x = tf.concat(1, embeddings)

            Y_pred_top_k = sess.run(self.pred_top_k, \
                feed_dict={self.X: tf_batch_x.eval()})

            # Y_pred_top_k[0][i]: K top values for token i (descending order)
            #
            # [ 12.58759022  11.70285416   6.45877123   6.01622534
            #   4.81124258 1.8947413    1.11786246  -0.2629841
            #   -1.64534914  -1.66685677]
            #
            # Y_pred_top_k[1][i]: K top indices for token i
            # [ 3  0  7 11 17 55 13  1  9 51]
            #
            # we're interested in indices (indicating our desired action)

            # change to major index being sentence
            pred_top_k_var = []

            for m in range(len(Y_pred_top_k[1])):
                assert len(Y_pred_top_k[1][m]) == topK
                pred_top_k_var.append(Y_pred_top_k[1][m])

            sentences = test_reader_decoded.getNextAnnotations()
            outputs.append(sentences)

        # also print out number of correct actions (even if tag was wrong)
        #logger.info('Correct actions+arcs: %d/%d, correct actions: %d' % \
        #    (correctElems, totalElems, correctActions))

        # we need to make sure unseen words get mapped to unknown values when
        # we use an unseen corpus

        token_count = 0
        deprel_correct = 0
        head_correct = 0
        deprel_and_head_correct = 0

        for sentences in outputs:
            print('-'*20)
            for sentence in sentences:
                print('-'*20)
                #print([w for w in sentence.tokens])
                for w in sentence.tokens:
                    suffix = ''

                    gold_head = w.HEAD
                    gold_deprel = w.DEPREL

                    if gold_head == -1:
                        # FIXME: is this giving ROOT node free credit
                        # even when predicted wrongly?
                        gold_deprel = "ROOT"

                    if w.parsedHead == -1:
                        # make it simple
                        w.parsedLabel = "ROOT"

                    if w.parsedLabel == gold_deprel:
                        deprel_correct += 1
                    else:
                        suffix = 'L'

                    if w.parsedHead == gold_head:
                        head_correct += 1
                    else:
                        suffix += 'H'

                    if w.parsedLabel == gold_deprel and \
                            w.parsedHead == gold_head:
                        deprel_and_head_correct += 1
                        suffix = 'O' # both correct

                    token_count += 1

                    if w.parsedHead == -1:
                        logger.info('%-20s%-10s%-5d%-5s' % \
                            (w.FORM, 'ROOT', w.parsedHead, suffix))
                    else:
                        logger.info('%-20s%-10s%-5d%-5s' % \
                            (w.FORM, w.parsedLabel, w.parsedHead, suffix))

        # errors that accumulate (tokens are tested based on previous decoded
        # decisions, which could screw up shifting and arcing, etc)
        logger.info('accumulated error measure')
        logger.info('accuracy(UAS): %d/%d' % (head_correct, token_count))
        logger.info('accuracy(LAS): %d/%d' % (deprel_and_head_correct, \
            token_count))
        logger.info('accuracy(deprel correct): %d/%d' % (deprel_correct, \
            token_count))

    def doEvaluationAndPrintOutput(self, sess):
        batchSize = self.modelParams.cfg['batchSize']
        featureStrings = self.modelParams.cfg['featureStrings']
        topK = self.modelParams.cfg['topK']

        #if not self.testingCorpus:
        self.testingCorpus = ParsedConllFile()
        self.testingCorpus.read(open(self.modelParams.testingFile, 'r',
                                         encoding='utf-8').read())

        # evaluate sentence-wide accuracy by UAS and LAS
        # of course, token errors can accumulate and this is why sentence-wide
        # accuracy is lower than token-only accuracy given by doEvaluation()

        # batch size set at one temporarily
        test_reader_decoded = DecodedParseReader(self.testingCorpus, \
            batchSize, featureStrings, self.featureMaps, epoch_print=False)

        correctActions = 0
        correctElems = 0
        totalElems = 0

        outputs = []

        filled_count = 0
        pred_top_k_var = []

        test_runs = 200
        for i in range(test_runs):
            logger.debug('Evaluation(batch %d)' % i)
            test_reader_output = test_reader_decoded.nextFeatureBags(
                pred_top_k_var, filled_count)

            if test_reader_output[0] == None:
                logger.critical('Reader error')
                return

            features_major_types, features_output, epochs, \
                filled_count = test_reader_output

            embeddings = []

            # order of features will match the order of our arrays above
            for k in range(len(features_output)):
                embeddings.append(tf.nn.embedding_lookup(
                    self.featureEmbeddings[k], features_output[k]))
                embeddings[-1] = tf.reshape(embeddings[-1],
                    [-1, len(self.featureNames[k]) * \
                         self.featureEmbeddingSizes[k]])
            
            tf_batch_x = tf.concat(1, embeddings)

            Y_pred_top_k = sess.run(self.pred_top_k, \
                feed_dict={self.X: tf_batch_x.eval()})

            # Y_pred_top_k[0][i]: K top values for token i (descending order)
            #
            # [ 12.58759022  11.70285416   6.45877123   6.01622534
            #   4.81124258 1.8947413    1.11786246  -0.2629841
            #   -1.64534914  -1.66685677]
            #
            # Y_pred_top_k[1][i]: K top indices for token i
            # [ 3  0  7 11 17 55 13  1  9 51]
            #
            # we're interested in indices (indicating our desired action)

            # change to major index being sentence
            pred_top_k_var = []

            for m in range(len(Y_pred_top_k[1])):
                assert len(Y_pred_top_k[1][m]) == topK
                pred_top_k_var.append(Y_pred_top_k[1][m])

            sentences = test_reader_decoded.getNextAnnotations()
            outputs.append(sentences)

        # also print out number of correct actions (even if tag was wrong)
        #logger.info('Correct actions+arcs: %d/%d, correct actions: %d' % \
        #    (correctElems, totalElems, correctActions))

        # we need to make sure unseen words get mapped to unknown values when
        # we use an unseen corpus

        token_count = 0
        deprel_correct = 0
        head_correct = 0
        deprel_and_head_correct = 0

        for sentences in outputs:
            print('-'*20)
            for sentence in sentences:
                print('-'*20)
                #print([w for w in sentence.tokens])
                for w in sentence.tokens:
                    suffix = ''

                    gold_head = w.HEAD
                    gold_deprel = w.DEPREL
                    if gold_head == -1:
                        # ??? FIXME: unconditionally?
                        gold_deprel = "ROOT"

                    if w.parsedHead == -1:
                        w.parsedLabel = "ROOT" # make it simple

                    if w.parsedLabel == gold_deprel:
                        deprel_correct += 1
                    else:
                        suffix = 'L'

                    if w.parsedHead == gold_head:
                        head_correct += 1
                    else:
                        suffix += 'H'

                    if w.parsedLabel == gold_deprel and \
                            w.parsedHead == gold_head:
                        deprel_and_head_correct += 1
                        suffix = 'O' # both correct

                    token_count += 1

                    if w.parsedHead == -1:
                        logger.info('%-20s%-10s%-5d%-5s' % \
                            (w.FORM, 'ROOT', w.parsedHead, suffix))
                    else:
                        logger.info('%-20s%-10s%-5d%-5s' % \
                            (w.FORM, w.parsedLabel, w.parsedHead, suffix))

        # errors that accumulate (tokens are tested based on previous decoded
        # decisions, which could screw up shifting and arcing, etc)
        logger.info('Propagated Error Measure (test_set)')
        logger.info('Accuracy(UAS): %d/%d (%.2f%%)' % \
            (head_correct, token_count,
            100.0 * float(head_correct) / float(token_count)))
        logger.info('Accuracy(LAS): %d/%d (%.2f%%)' % \
            (deprel_and_head_correct, token_count,
            100.0 * float(deprel_and_head_correct) / float(token_count)))
        logger.info('Accuracy(DepRel): %d/%d' % \
            (deprel_correct, token_count,
            100.0 * float(deprel_correct) / float(token_count)))

modelParams = ModelParameters(args.model_folder)
modelParams.trainingFile = args.training_file
modelParams.testingFile = args.testing_file

# set variables from parser-config and isolate them in a separate namespace
# to avoid collisions with this code
configFile = open(modelParams.getFilePath('parser-config'), 'r', \
                  encoding='utf-8').read()
compile(configFile, '<string>', 'exec')
configNamespace = {}
exec(configFile, configNamespace)

# TODO: use trainingSteps
requiredFields = ['trainingSteps', 'learningRate', 'batchSize', 'topK',
                  'hiddenLayerSizes', 'embeddingSizes', 'featureStrings']
for field in requiredFields:
    assert configNamespace[field] != None, 'please set %s in config' % field

modelParams.cfg = configNamespace
modelParams.lexicon = Lexicon(modelParams)

if args.train:
    parser = Parser(modelParams, 'train')
    parser.startTraining()
else:
    assert None, 'evaluation mode not implemented'
