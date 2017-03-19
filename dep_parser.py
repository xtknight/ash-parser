from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
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
import json
import random

from model_parameters import *
from lexicon import *
from utils import *
from conll_utils import *
from feature_extractor import SparseFeatureExtractor
from parser_state import ParserState
from arc_standard_transition_system import ArcStandardTransitionState, \
     ArcStandardTransitionSystem
from arc_eager_transition_system import ArcEagerTransitionState, \
     ArcEagerTransitionSystem
from gold_parse_reader import GoldParseReader
from decoded_parse_reader import DecodedParseReader
from tensorflow.python.ops import state_ops

logger = logging.getLogger('DepParser')

parser = argparse.ArgumentParser(
    description='Train a Chen and Manning-style neural network dependency' \
                ' parser')

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
parser.add_argument('--restart', action='store_true', default=False,
                    help='Re-train model from scratch instead of restoring '
                         'a previously saved model')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to run (run-throughs over all '
                         'training corpus feature bags). Default 10')
parser.add_argument('--scoring-strategy', type=str, default='default',
                    help='Choices: "default", "conllx", "ignore_parens"')
#parser.add_argument('--feature-bag', type=str,
#                    help='Specify pre-created feature bag file to save' \
#                         ' computation time (saved in model dir by default')
#parser.add_argument('--epochs', type=int, default=10,
#                    help='Training epochs (default 10). Shuffle sentences '
#                         ' and re-train during each training epoch.')

## TODO:
# add param: use pretrained word/sense embeddings gensim/Mikolov

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

if not (args.scoring_strategy == 'default' or \
        args.scoring_strategy == 'conllx' or \
        args.scoring_strategy == 'ignore_parens'):
    print('Unknown scoring strategy "%s"' % args.scoring_strategy)
    sys.exit(1)

def batchedSparseToDense(sparse_indices, output_size):
    """Batch compatible sparse to dense conversion.

    This is useful for one-hot coded target labels.

    Args:
      sparse_indices: [batch_size] tensor containing one index per batch
      output_size: needed in order to generate the correct dense output

    Returns:
      A [batch_size, output_size] dense tensor.
    """
    eye = tf.diag(tf.fill([output_size], tf.constant(1, tf.float32)))
    return tf.nn.embedding_lookup(eye, sparse_indices)

def embeddingLookupFeatures(params, ids):
    """Computes embeddings for each entry of sparse features sparse_features.

    Args:
      params: list of 2D tensors containing vector embeddings
      sparse_features: 1D tensor of strings. Each entry is a string encoding of
        dist_belief.SparseFeatures, and represents a variable length list of
        feature ids, and optionally, corresponding weights values.
      allow_weights: boolean to control whether the weights returned from the
        SparseFeatures are used to multiply the embeddings.

    Returns:
      A tensor representing the combined embeddings for the sparse features.
      For each entry s in sparse_features, the function looks up the embeddings
      for each id and sums them into a single tensor weighing them by the
      weight of each id. It returns a tensor with each entry of sparse_features
      replaced by this combined embedding.
    """
    #params = tensorDumpValsAllIter(params, [params], '/tmp/ash_params')

    if not isinstance(params, list):
      params = [params]

    # Lookup embeddings.
    embeddings = tf.nn.embedding_lookup(params, ids)
    '''
    embeddings.shape (6144, 32)
    embeddings.shape (10240, 32)
    embeddings.shape (10240, 64)
    '''
    #embeddings = tensorDumpValsAllIter(embeddings, [embeddings],
    #    '/tmp/ash_embeddings')
    return embeddings

'''
Takes an SHA-1 hash of a file
(Useful for hashing training corpus)
'''
def fileHash(fname):
    fd = open(fname, 'rb')
    retval = hashlib.sha1(fd.read()).hexdigest()
    fd.close()
    return retval

'''
Entry point for dependency parser
'''
class Parser(object):
    def __init__(self, modelParams):
        self.logger = logging.getLogger('Parser')
        self.modelParams = modelParams

        self.variables = {}
        self.params = {}
        self.trainableParams = []
        self.inits = {}
        self.averaging = {}
        self.averaging_decay = self.modelParams.cfg['averagingDecay']
        self.use_averaging = True
        self.check_parameters = True
        self.training = {}
        self.evaluation = {}

        with tf.name_scope('params') as self._param_scope:
            pass

        #self.trainingCorpus = None
        #self.testingCorpus = None

    def getStep(self):
        def onesInitializer(shape, dtype=tf.float32, partition_info=None):
            return tf.ones(shape, dtype)
        return self.addVariable([], tf.int32, 'step', onesInitializer)

    def incrementCounter(self, counter):
        return state_ops.assign_add(counter, 1, use_locking=True)

    def addLearningRate(self, initial_learning_rate, decay_steps):
        """Returns a learning rate that decays by 0.96 every decay_steps.

        Args:
            initial_learning_rate: initial value of the learning rate
            decay_steps: decay by 0.96 every this many steps

        Returns:
            learning rate variable.
        """
        step = self.getStep()
        return cf.with_dependencies(
            [self.incrementCounter(step)],
            tf.train.exponential_decay(initial_learning_rate,
                step,
                decay_steps,
                0.96,
                staircase=True))

    def addVariable(self, shape, dtype, name, initializer=None):
        if name in self.variables:
            return self.variables[name]
        self.variables[name] = tf.get_variable(name, shape, dtype, initializer)
        if initializer is not None:
            self.inits[name] = state_ops.init_variable(self.variables[name],
                               initializer)
        return self.variables[name]

    '''
    Don't use variable_scope, as param names will overwrite each other
    '''
    def addParam(self, shape, dtype, name, initializer=None,
            return_average=False):
        # this isn't a problem. we reload variables if they already exist.        
        #if name in self.params:
        #    self.logger.warning(name + ' already exists!')

        if name not in self.params:
            step = tf.cast(self.getStep(), tf.float32)
            with tf.name_scope(self._param_scope):
                # Put all parameters and their initializing ops in their own
                # scope irrespective of the current scope (training or eval).
                self.params[name] = tf.get_variable(name, shape, dtype,
                    initializer)
                param = self.params[name]

                if initializer is not None:
                    self.inits[name] = state_ops.init_variable(param,
                        initializer)
                if self.averaging_decay == 1:
                    self.logging.info('Using vanilla averaging of parameters.')
                    ema = tf.train.ExponentialMovingAverage(
                        decay=(step / (step + 1.0)), num_updates=None)
                else:
                    ema = tf.train.ExponentialMovingAverage(
                        decay=self.averaging_decay, num_updates=step)

                self.averaging[name + '_avg_update'] = ema.apply([param])
                self.variables[name + '_avg_var'] = ema.average(param)
                self.inits[name + '_avg_init'] = state_ops.init_variable(
                        ema.average(param), tf.zeros_initializer())
        return (self.variables[name + '_avg_var'] if return_average else
                        self.params[name])

    def addEmbedding(self, features, num_features, num_ids, embedding_size,
            major_type, return_average=False):
        initializer = tf.random_normal_initializer(
                    stddev=1.0 / embedding_size**.5, \
                    seed=0)

        embedding_matrix = self.addParam(
            [num_ids, embedding_size],
            tf.float32,
            'embedding_matrix_%s' % major_type,
            initializer,
            return_average=return_average)

        embedding = embeddingLookupFeatures(embedding_matrix,
                                            tf.reshape(features,
                                                [-1],
                                                name='feature_%s' % major_type))

        return tf.reshape(embedding, [-1, num_features * embedding_size])

    '''
    Setup transition and action system and feature maps
    (necessary whether training or evaluating)
    '''
    def setupParser(self, mode):
        hiddenLayerSizes = self.modelParams.cfg['hiddenLayerSizes']
        featureStrings = self.modelParams.cfg['featureStrings']
        embeddingSizes = self.modelParams.cfg['embeddingSizes']
        batchSize = self.modelParams.cfg['batchSize']

        #self.transitionSystem = ArcStandardTransitionSystem()
        self.transitionSystem = ArcEagerTransitionSystem()

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
            .extract(ParserState(ParsedConllSentence(docid=None),
                     self.featureMaps), doLogging=False)

        featureTypeInstances = fvec.types
        self.featureMajorTypeGroups, _ = fvec.concatenateSimilarTypes()

        # index: major feature type index
        # values: feature names under that type
        self.featureNames = [[] for t in self.featureMajorTypeGroups]

        self.logger.info('Detected major feature groups (in alphabetical '
            'order): ' + str(self.featureMajorTypeGroups))

        self.featureDomainSizes = []
        #self.featureEmbeddings = []

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

        assert len(self.featureDomainSizes) == len(self.featureEmbeddingSizes)
        #assert len(self.featureDomainSizes) == len(self.featureEmbeddings)
        assert len(self.featureDomainSizes) == len(self.featureNames)

        self.logger.info('')
        self.logger.info('Batch size (number of parser states): %d' % batchSize)
        self.logger.info('Total feature count: %d' % \
            (len(featureTypeInstances)))
        self.logger.info('Total bag of features length per state: %d' % \
            (self.BAG_OF_FEATURES_LEN))
        self.logger.info('Total features input size: %d' % \
            (batchSize*self.BAG_OF_FEATURES_LEN))

        # for actions, we don't encode UNKNOWN, ROOT, or OUTSIDE
        # we only encode the number of base values
        self.ACTION_COUNT = self.transitionSystem.numActions(
            self.featureMaps['label'].lastBaseValue + 1)

        self.logger.info('Total action count: %d' % self.ACTION_COUNT)

    '''
    Setup TensorFlow Variables in model
    '''
    def buildNetwork(self, mode='train'):
        assert mode == 'train' or mode == 'eval'

        if mode == 'train':
            return_average = False
            nodes = self.training
        else:
            return_average = self.use_averaging
            nodes = self.evaluation

        learningRate = self.modelParams.cfg['learningRate']
        decaySteps = self.modelParams.cfg['decaySteps']
        # FIXME: does momentum/learning rate reload properly when retraining?
        momentum = self.modelParams.cfg['momentum']
        topK = self.modelParams.cfg['topK']
        hiddenLayerSizes = self.modelParams.cfg['hiddenLayerSizes']
        batchSize = self.modelParams.cfg['batchSize']

        with tf.name_scope(mode):
            weights = []
            biases = []
            embeddings = []
            nodes['feature_endpoints'] = []

            for i in range(len(self.featureMajorTypeGroups)):
                major_type = self.featureMajorTypeGroups[i]
                # shape will be [-1, number of sparse integer features in group]
                nodes['feature_endpoints'].append(tf.placeholder(tf.int32, \
                    [None, len(self.featureNames[i])],
                    name="ph_feature_endpoints_%s" % major_type))
                embeddings.append(self.addEmbedding( \
                                            nodes['feature_endpoints'][i],
                                            len(self.featureNames[i]),
                                            self.featureDomainSizes[i],
                                            self.featureEmbeddingSizes[i],
                                            major_type,
                                            return_average=return_average))

            # Input layer
            last_layer = tf.concat(embeddings, 1)
            last_layer_size = self.BAG_OF_FEATURES_LEN

            # Hidden layers
            for i in range(len(hiddenLayerSizes)):
                h = hiddenLayerSizes[i]

                weights.append(self.addParam(
                    [last_layer_size, h],
                    tf.float32,
                    'layer_%d_weights' % i,
                    tf.random_normal_initializer(stddev=1e-4, seed=0),
                    return_average=return_average))

                biases.append(self.addParam(
                    [h],
                    tf.float32,
                    'layer_%d_biases' % i,
                    tf.constant_initializer(0.2),
                    return_average=return_average))

                last_layer = tf.nn.relu_layer(last_layer,
                                    weights[-1],
                                    biases[-1],
                                    name='layer_%d' % i)
                last_layer_size = h
            
            # Output layer
            weights.append(self.addParam(
                [last_layer_size, self.ACTION_COUNT],
                tf.float32,
                'softmax_weights',
                tf.random_normal_initializer(stddev=1e-4, seed=0),
                return_average=return_average))

            biases.append(self.addParam(
                [self.ACTION_COUNT],
                tf.float32,
                'softmax_biases',
                tf.zeros_initializer(),
                return_average=return_average))
            
            logits = tf.nn.xw_plus_b(last_layer,
                             weights[-1],
                             biases[-1],
                             name='logits')

            if mode == 'train':
                nodes['gold_actions'] = tf.placeholder(tf.int32, [None], \
                    name='ph_gold_actions')
                nodes['filled_slots'] = tf.placeholder(tf.int32, \
                    name='ph_filled_slots')

                dense_golden = batchedSparseToDense(nodes['gold_actions'], \
                    self.ACTION_COUNT)

                cross_entropy = tf.div(
                    tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                            logits=logits, labels=dense_golden)),
                        tf.cast(nodes['filled_slots'], tf.float32))

                # regularize all parameters except output layer
                regularized_params = [tf.nn.l2_loss(p) for p in weights[:-1]]
                regularized_params += [tf.nn.l2_loss(p) for p in biases[:-1]]

                l2_loss = 1e-4 * tf.add_n(regularized_params) \
                    if regularized_params else 0

                cost = tf.add(cross_entropy, l2_loss, name='cost')

                lr = self.addLearningRate(learningRate, decaySteps)

                optimizer = tf.train.MomentumOptimizer(lr,
                    momentum,
                    use_locking=False)

                trainableParams = self.params.values()

                train_op = optimizer.minimize(cost, var_list=trainableParams)

                for param in trainableParams:
                    slot = optimizer.get_slot(param, 'momentum')
                    self.inits[slot.name] = state_ops.init_variable(slot,
                        tf.zeros_initializer())
                    self.variables[slot.name] = slot

                numerical_checks = [
                    tf.check_numerics(param,
                        message='Parameter is not finite.')
                    for param in trainableParams
                    if param.dtype.base_dtype in [tf.float32, tf.float64]
                ]
                check_op = tf.group(*numerical_checks)
                avg_update_op = tf.group(*self.averaging.values())
                train_ops = [train_op]
                if self.check_parameters:
                    train_ops.append(check_op)
                if self.use_averaging:
                    train_ops.append(avg_update_op)

                nodes['train_op'] = tf.group(*train_ops, name='train_op')
                nodes['cost'] = cost
                nodes['logits'] = logits
            else:
                nodes['logits'] = logits

    '''
    Serialize the feature definitions
    (so that we can determine when they change)
    '''
    def serializeFeatureDef(self):
        d = []

        bs = self.modelParams.cfg['batchSize']
        d.append(bs)

        # if projectivize parameter is changed, we may have to recalculate
        # features as well (in case there are non-projective sentences)
        p = self.modelParams.cfg['projectivizeTrainingSet']
        d.append(p)

        fs = self.modelParams.cfg['featureStrings']
        # order doesn't matter
        fs.sort()
        d.append(fs)

        e = []
        # because dictionaries aren't ordered...
        for (k, v) in self.modelParams.cfg['embeddingSizes'].items():
            e.append((k,v))
        # sort by key
        e.sort()

        d.append(e)
        return json.dumps(d)

    '''
    Generate or load pre-computed feature bags
    '''
    def obtainFeatureBags(self, trainingFileName):
        batchSize = self.modelParams.cfg['batchSize']
        projectivizeTrainingSet = self.modelParams.cfg \
            ['projectivizeTrainingSet']

        activeFeatureDef = self.serializeFeatureDef().strip()
        activeCorpusHash = fileHash(trainingFileName)

        cachedFeatureDef = None
        try:
            fd = open(self.modelParams.getFilePath('feature-def'), 'r',
                encoding='utf-8')
            cachedFeatureDef = fd.read().strip()
            fd.close()
        except:
            cachedFeatureDef = None

        cachedCorpusHash = None
        try:
            fd = open(self.modelParams.getFilePath('training-corpus-hash'), 'r',
                encoding='utf-8')
            cachedCorpusHash = fd.read().strip()
            self.logger.debug('Cached corpus hash: %s' % cachedCorpusHash)
            fd.close()
        except:
            cachedCorpusHash = None

        self.logger.debug('Training corpus hash: %s' % activeCorpusHash)
        self.logger.debug('Cached corpus hash: %s' % cachedCorpusHash)

        self.logger.debug('Active feature definition: %s' % activeFeatureDef)
        self.logger.debug('Cached feature definition: %s' % cachedFeatureDef)

        if activeFeatureDef == cachedFeatureDef and \
                activeCorpusHash == cachedCorpusHash:
            self.logger.info('Loading pre-existing feature bags...')
            fd = open(self.modelParams.getFilePath('feature-bag-bin'), 'rb')
            batches = pickle.load(fd)
            fd.close()
        else:
            featureStrings = self.modelParams.cfg['featureStrings']
            self.logger.info('Feature bag needs recalculation (first training' \
                ' or features changed)')

            # parameters here must match parameters during lexicon generation
            trainingCorpus = ParsedConllFile(keepMalformed=False,
                projectivize=projectivizeTrainingSet)

            trainingCorpus.read(open(self.modelParams.trainingFile, 'r',
                encoding='utf-8').read())

            # Start getting sentence batches...
            reader = GoldParseReader(trainingCorpus, batchSize, \
                featureStrings, self.featureMaps, epoch_print=False)

            batches = []

            i = 0
            while(True):
                self.logger.info('Generating feature bag #%d...' % (i+1))
                reader_output = reader.nextFeatureBags()
                if reader_output[0] == None:
                    self.logger.debug('Iter(%d): reader output is None' % i)
                    break

                features_major_types, features_output, gold_actions, \
                    epoch_num = reader_output

                if epoch_num > 1:
                    # don't make more than one epoch
                    break

                batches.append(reader_output)
                i += 1

            self.logger.info('Saving feature bags...')

            fd = open(self.modelParams.getFilePath('feature-bag-bin'), 'wb')
            pickle.dump(batches, fd)
            fd.close()

            fd = open(self.modelParams.getFilePath('feature-def'), 'w',
                encoding='utf-8')
            fd.write(activeFeatureDef)
            fd.close()

            fd = open(self.modelParams.getFilePath('training-corpus-hash'), 'w',
                encoding='utf-8')
            fd.write(activeCorpusHash)
            fd.close()

        return batches

    '''
    Start training from scratch, or from where we left off
    '''
    def startTraining(self, sess, epochs_to_run=10, restart=False):
        batchSize = self.modelParams.cfg['batchSize']
        featureStrings = self.modelParams.cfg['featureStrings']

        ckpt_dir = fixPath(self.modelParams.modelFolder) + '/'
        saver = tf.train.Saver()

        if restart:
            self.logger.info('Start fitting')
        else:
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restore variables from disk.
                saver.restore(sess, ckpt.model_checkpoint_path)
                self.logger.info('Model restored')
                self.logger.info('Continue fitting')
            else:
                self.logger.info('Start fitting')

        print_freq = 10

        save_freq = 500
        #eval_freq = 200

        batches = self.obtainFeatureBags(self.modelParams.trainingFile)

        epoch_num = 0

        while epoch_num < epochs_to_run:
            i = 0
            while i < len(batches):
                reader_output = batches[i]
                if reader_output[0] == None:
                    self.logger.debug('Iter(%d): reader output is None' % i)
                    break

                '''
                    epoch_num refers to the number of run-throughs through the
                    whole training corpus, whereas `i` is just the batch
                    iteration number
                '''

                features_major_types, features_output, gold_actions, \
                    _ = reader_output

                filled_count = len(gold_actions)
                if filled_count < batchSize:
                    # break out (partial batches seem to completely ruin the
                    # model for whatever reason)
                    # use continue because in case we shuffle the outer
                    # dimension, we might get the partial batches in the
                    # middle
                    i += 1
                    continue
                
                #print('feature(0) len: %d' % len(features_output[0]))
                #print('feature(1) len: %d' % len(features_output[1]))
                #print('feature(2) len: %d' % len(features_output[2]))

                # debug: print out first 40 actions (useful to compare with
                #        SyntaxNet)
                self.logger.debug('gold_actions: %s' % \
                    str(gold_actions[:40]))

                assert len(self.training['feature_endpoints']) == \
                    len(features_output)

                feed_dict = {}
                for k in range(len(self.training['feature_endpoints'])):
                    features_output[k] = np.asarray(features_output[k])
                    feed_dict[self.training['feature_endpoints'][k]] = \
                        features_output[k].reshape( \
                            [-1, len(self.featureNames[k])])

                feed_dict[self.training['filled_slots']] = filled_count
                feed_dict[self.training['gold_actions']] = gold_actions

                c, _ = sess.run([self.training['cost'],
                                self.training['train_op']],
                                feed_dict=feed_dict)

                if i > 0 and i % print_freq == 0:
                    self.logger.info('Epoch: %04d Iter: %06d cost=%s' % \
                        (epoch_num+1, i+1, "{:.2f}".format(c)))
                    #self.quickEvaluationMetric(sess, mode='training')
                    # reset avg
                    #avg_cost = 0.0

                if i > 0 and i % save_freq == 0:
                    save_path = saver.save(sess, ckpt_dir + 'model.ckpt')
                    self.logger.info('Model saved to file: %s' % save_path)
                    #self.attachmentMetric(sess, runs=100, mode='training')
                    #self.attachmentMetric(sess, runs=100, mode='testing')

                #if i > 0 and i % eval_freq == 0:
                #    self.attachmentMetric(sess, runs=200)

                i += 1

            epoch_num += 1

            if epoch_num < epochs_to_run:
                # evaluate now. otherwise evaluate after training
                # complete message is shown
                #self.attachmentMetric(sess, runs=100, mode='training')
                pass
            else:
                self.logger.info('Training is complete (%d epochs)' % \
                    epochs_to_run)
                save_path = saver.save(sess, ckpt_dir + 'model.ckpt')
                self.logger.info('Model saved to file: %s' % save_path)
                self.attachmentMetric(sess, runs=100, mode='testing')
                return

    def attachmentMetric(self, sess, runs=200, mode='testing'):
        batchSize = self.modelParams.cfg['batchSize']

        #batchSize = 128 # let's try a smaller batch for evaluation
        featureStrings = self.modelParams.cfg['featureStrings']
        topK = self.modelParams.cfg['topK']

        assert mode == 'testing' or mode == 'training'

        testingCorpus = ParsedConllFile()
        if mode == 'testing':
            testingCorpus.read(open(self.modelParams.testingFile, 'r',
                                     encoding='utf-8').read())
        elif mode == 'training':
            testingCorpus.read(open(self.modelParams.trainingFile, 'r',
                                     encoding='utf-8').read())

        # evaluate sentence-wide accuracy by UAS and LAS
        # of course, token errors can accumulate and this is why sentence-wide
        # accuracy is lower than token-only accuracy given by
        # quickEvaluationMetric()

        # batch size set at one temporarily
        test_reader_decoded = DecodedParseReader(testingCorpus, \
            batchSize, featureStrings, self.featureMaps, epoch_print=False)

        correctActions = 0
        correctElems = 0
        totalElems = 0

        outputs = []

        filled_count = 0

        self.buildNetwork('eval')

        # eventually will be (filled_count, num_actions)
        logits = np.asarray([])

        test_runs = runs
        for i in range(test_runs):
            logger.debug('Evaluation(batch %d)' % i)
            test_reader_output = test_reader_decoded.nextFeatureBags(
                logits, filled_count)

            if test_reader_output[0] == None:
                logger.critical('Reader error')
                return

            features_major_types, features_output, epochs, \
                filled_count = test_reader_output

            assert len(self.evaluation['feature_endpoints']) == \
                len(features_output)

            feed_dict = {}
            for k in range(len(self.evaluation['feature_endpoints'])):
                features_output[k] = np.asarray(features_output[k])
                feed_dict[self.evaluation['feature_endpoints'][k]] = \
                    features_output[k].reshape( \
                        [-1, len(self.featureNames[k])])

            logits = sess.run(self.evaluation['logits'], feed_dict=feed_dict)
            logits = np.asarray(logits)

            logger.info('Evaluating batch %d/%d...' % (i+1, test_runs))

            sentences = test_reader_decoded.getNextAnnotations()
            outputs.append(sentences)

        token_count = 0
        deprel_correct = 0
        head_correct = 0
        deprel_and_head_correct = 0

        for sentences in outputs:
            logger.info('-'*20)
            for sentence in sentences:
                logger.info('-'*20)
                #logger.info([w for w in sentence.tokens])
                for w in sentence.tokens:
                    suffix = ''

                    gold_head = w.HEAD
                    gold_deprel = w.DEPREL
                    if gold_head == -1:
                        gold_deprel = 'ROOT'

                    if w.parsedHead == -1:
                        # make it simple
                        w.parsedLabel = 'ROOT'

                    if shouldScoreToken(w.FORM, w.UPOSTAG,
                            self.modelParams.scoring_strategy):
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
                            # mark both correct
                            suffix = 'O'

                        token_count += 1

                        if w.parsedHead == -1:
                            logger.info('%-20s%-10s%-5d%-5s' % \
                                (w.FORM, 'ROOT', w.parsedHead, suffix))
                        else:
                            logger.info('%-20s%-10s%-5d%-5s' % \
                                (w.FORM, w.parsedLabel, w.parsedHead, suffix))
                    else:
                        logger.debug('Not scoring token: form="%s", tag="%s"' \
                            % (w.FORM, w.UPOSTAG))

        if token_count <= 0:
            logger.warning('No tokens to calculate Attachment Error Metric')
            return

        # errors that accumulate (tokens are tested based on previous decoded
        # decisions, which could screw up shifting and arcing, etc)
        # SyntaxNet uses UAS (HEAD-only) for its evaluation during training!
        logger.info('Attachment Error Metric (%s_set)' % mode)
        logger.info('Scoring Strategy: %s' % \
            self.modelParams.scoring_strategy)
        logger.info('Accuracy(UAS): %d/%d (%.2f%%)' % \
            (head_correct, token_count,
            100.0 * float(head_correct) / float(token_count)))
        logger.info('Accuracy(LAS): %d/%d (%.2f%%)' % \
            (deprel_and_head_correct, token_count,
            100.0 * float(deprel_and_head_correct) / float(token_count)))
        logger.info('Accuracy(DepRel): %d/%d (%.2f%%)' % \
            (deprel_correct, token_count,
            100.0 * float(deprel_correct) / float(token_count)))

def __main__():
    modelParams = ModelParameters(args.model_folder)
    modelParams.trainingFile = args.training_file
    modelParams.testingFile = args.testing_file

    # set variables from parser-config and isolate them in a separate namespace
    # to avoid collisions with this code
    fd = open(modelParams.getFilePath('parser-config'), 'r', \
                      encoding='utf-8')
    configFile = fd.read()
    fd.close()

    fd = open(modelParams.getFilePath('trained-config'), 'w', \
                      encoding='utf-8')
    fd.write(configFile)
    fd.close()

    compile(configFile, '<string>', 'exec')
    configNamespace = {}
    exec(configFile, configNamespace)
    
    requiredFields = ['learningRate', 'batchSize', 'topK',
                      'hiddenLayerSizes', 'embeddingSizes', 'featureStrings',
                      'momentum', 'projectivizeTrainingSet']
    for field in requiredFields:
        assert configNamespace[field] != None, 'please set %s in config' % field

    modelParams.cfg = configNamespace
    modelParams.lexicon = Lexicon(modelParams)
    modelParams.scoring_strategy = args.scoring_strategy

    if args.train:
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth=True
        #config.gpu_options.per_process_gpu_memory_fraction=1.0

        # very important for Parser to be under a session scope
        with tf.Session(config=config) as sess:
            parser = Parser(modelParams)
            #print(parser.inits.values())

            # perform variable initialization

            parser.setupParser('train')
            parser.buildNetwork('train')
            sess.run(list(parser.inits.values()))

            writer = tf.summary.FileWriter(modelParams.modelFolder, \
                graph=tf.get_default_graph())

            parser.startTraining(sess, epochs_to_run=args.epochs,
                restart=args.restart)
    else:
        assert None, 'evaluation mode not implemented'

__main__()
