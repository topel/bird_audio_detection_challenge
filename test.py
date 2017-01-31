import theano
import theano.tensor as T

from model_utils import *
from output_utils import confusion_matrix, save_predictions

from sklearn.metrics import roc_auc_score, roc_curve

def main(train_corpus, corpusname, test_set, modeldir, options, nb_samples, csvfile, model, modelfn, loss_type):

    NB_CHANNELS, NB_FRAMES, NB_FEATURES, NB_CLASSES, BATCH_SIZE, removeMean, divideStd, TEST_LABELS, doAugment, feature_type = options['NB_CHANNELS'], \
                                                                           options['NB_FRAMES'], \
                                                                           options['NB_FEATURES'], \
                                                                           options['NB_CLASSES'], \
                                                                           options['BATCH_SIZE'],\
                                                                           options['CENTER_DATA'], \
                                                                           options['REDUCE_DATA'], \
                                                                           options['TEST_LABELS'], \
                                                                           options['AUGMENT'], \
                                                                           options['FEATURE_TYPE']

    print 'OPTIONS: ', options
    # Prepare Theano variables for inputs and targets
    # input_var = T.tensor4('inputs', dtype='float32')
    if feature_type == 'ivec':
        input_var = T.tensor3('inputs')
    else:
        input_var = T.tensor4('inputs')

    # utiliser des int8 ne marche pas:
    # target_var = T.bvector('targets')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    # use batchnorm?
    useBN = True
    if model == 'densenet':
        if feature_type == 'fbank'  or feature_type == 'slicedfft' or feature_type == 'fbank_d_dd' or feature_type == 'fp' or feature_type == 'fp3':
            network, input_layer, output_layer_1 = build_densenet(input_shape=(None, NB_CHANNELS, NB_FRAMES, NB_FEATURES), input_var=input_var, classes=NB_CLASSES,
                       depth=19, first_output=32, growth_rate=15, num_blocks=3,
                       dropout=0, feature_type=feature_type)
        if feature_type == 'slicedfft':
            network, input_layer, output_layer_1 = build_densenet(input_shape=(None, NB_CHANNELS, NB_FRAMES, NB_FEATURES), input_var=input_var, classes=NB_CLASSES,
                       depth=16, first_output=32, growth_rate=20, num_blocks=3,
                       dropout=0, feature_type=feature_type)
        elif feature_type == 'fft':
            network, input_layer, output_layer_1 = build_densenet(input_shape=(None, NB_CHANNELS, NB_FRAMES, NB_FEATURES), input_var=input_var, classes=NB_CLASSES,
                       depth=11, first_output=32, growth_rate=20, num_blocks=2,
                       dropout=0, feature_type=feature_type)
        elif feature_type == 'mfcc':
            network, input_layer, output_layer_1 = build_densenet(input_shape=(None, NB_CHANNELS, NB_FRAMES, NB_FEATURES), input_var=input_var, classes=NB_CLASSES,
                       depth=19, first_output=32, growth_rate=15, num_blocks=3,
                       dropout=0, feature_type=feature_type)
    else:
        print("Unrecognized model type %r." % model)

    print("Loading model...")

    with np.load(modelfn) as f:
        single_array = [f['arr_%d' % i] for i in range(len(f.files))]
        param_values = [el for el in single_array[0]]
    lasagne.layers.set_all_param_values(network, param_values)

    print 'INFO: total number of layers:', len(lasagne.layers.get_all_layers(network))
    print("INFO: number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))

    probas = lasagne.layers.get_output(network, deterministic=True)

    # print probas.type
    # TensorType(float32, matrix)

    if loss_type == 'categorical_crossentropy':
        loss = lasagne.objectives.categorical_crossentropy(probas, target_var)
        # As a bonus, also create an expression for the classification accuracy:
        predictions = T.argmax(probas, axis=1)
        acc = T.mean(T.eq(predictions, target_var),
                          dtype=theano.config.floatX)
        # other option
        # acc = lasagne.objectives.categorical_accuracy(probas, target_var)
    elif loss_type == 'binary_hinge':
        threshold = 0.5  # prob min to get the positive class
        probas_vector = T.extra_ops.squeeze(probas) # probas type is col, need a vector to work properly
        predictions = theano.tensor.ge(probas_vector, threshold)
        loss = lasagne.objectives.binary_hinge_loss(probas_vector, target_var)
        acc = lasagne.objectives.binary_accuracy(probas_vector, target_var)
    elif loss_type == 'binary_crossentropy':
        threshold = 0.5  # prob min to get the positive class
        # print probas.type, T.extra_ops.squeeze(probas).type, theano.tensor.ge(T.extra_ops.squeeze(probas), threshold)
        probas_vector = T.extra_ops.squeeze(probas) # probas type is col, need a vector to work properly
        predictions = theano.tensor.ge(probas_vector, threshold)
        # loss = lasagne.objectives.binary_crossentropy(probas, target_var) # does not give the same results as using probas_vector
        loss = theano.tensor.nnet.binary_crossentropy(probas_vector, target_var)
        acc = lasagne.objectives.binary_accuracy(probas_vector, target_var)
        # acc = theano.tensor.eq(predictions, target_var)
    elif loss_type == 'weighted_binary_crossentropy':
        threshold = 0.5  # prob min to get the positive class
        # print test_probas.type, T.extra_ops.squeeze(test_probas).type, theano.tensor.ge(T.extra_ops.squeeze(test_probas), threshold)
        probas_vector = T.extra_ops.squeeze(probas) # test_probas type is col, need a vector to work properly
        predictions = theano.tensor.ge(probas_vector, threshold)
        if corpus == 'ff1010bird':
            w_pos=1935.0/(1935.0 + 5755.0) # nb of pos examples / nb of examples
        else:
            raise Exception('Please define a weight!! L183')
        loss = weighted_binary_crossentropy(probas_vector, target_var, w_pos=w_pos)
        acc = lasagne.objectives.binary_accuracy(probas_vector, target_var)


    loss = loss.mean()
    acc = acc.mean()

    train_mean = None
    train_std = None

    if TEST_LABELS:
        test_fn = theano.function([input_var, target_var], [loss, acc, probas, predictions])
        test_err, acc, test_batches, test_pred_probs, test_pred, test_gt = test_model(test_set, train_mean, train_std, removeMean, divideStd,
                                                                                           NB_CLASSES, feature_type, test_fn)
    else:
        test_fn = theano.function([input_var], [probas, predictions])
        test_pred_probs, test_pred = test_model_nolabels(test_set, train_mean, train_std, removeMean, divideStd, NB_CLASSES, feature_type, test_fn)

    assert test_pred.shape[0] == nb_samples, "ERROR: pred shape != nb samples"

    print feature_type, test_pred.shape, test_pred_probs.shape, nb_samples

    # save probas
    outfile=modeldir + '/%s_%s_%s_%s_%s_%s_probs.csv'%(train_corpus, 'Train', corpusname, subset, feature_type, kwargs['model'])
    # print 'DEBUG', csvfile
    # print test_pred_probs[:10], test_pred[:10]

    save_predictions(csvfile, loss_type, test_pred_probs, test_pred, outfile)

    if corpus != 'bad2016test':
        if NB_CLASSES > 1:
            # plot roc curve for test: class 1 only
            fpr, tpr, thresholds = roc_curve(test_gt, test_pred_probs[:, 1])
            test_roc_auc_score = roc_auc_score(test_gt, test_pred_probs[:,1], average='macro')
        else:
            # plot roc curve for test
            fpr, tpr, thresholds = roc_curve(test_gt, test_pred_probs)
            test_roc_auc_score = roc_auc_score(test_gt, test_pred_probs, average='macro')

        # print test_pred[:20], test_gt[:20]
        print("test loss: {:.6f}\t\t acc: {:.2f} %\t\t auc: {:.2f}\t\t nb_batches: {:.2f}".format(test_err / test_batches,
                                                                             100. * acc / test_batches,
                                                                             100. * test_roc_auc_score,
                                                                            test_batches))


        LABELS = {'0': 0, '1': 1}
        cm, cm_normalized =confusion_matrix(LABELS=LABELS, y_test=test_gt, test_pred=test_pred)

        print '%.2f / %.2f - %.1f - %.1f'%(100. * test_roc_auc_score, 100. * acc / test_batches, 100. * cm_normalized[0,0], 100. * cm_normalized[1,1])

        from os.path import dirname
        import cPickle as pickle
        modeldir = dirname(modelfn)
        pickle.dump({'fpr': fpr, 'tpr': tpr}, open(modeldir + '/fpr_tpr_%s_%s.txt'%(corpusname, subset),  'w'))

        print 'INFO: %s SAVED'%(modeldir + '/fpr_tpr_%s_%s.txt'%(corpusname, subset))

        # fpr_th05 = fpr[30]
        # tpr_th05 = tpr[30]

        # for ind, el in enumerate(thresholds):
        #     print ind, el

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                               lw=lw, label='ROC curve (area = %0.2f)' %(test_roc_auc_score))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('plots/roc_%s.png'%corpusname)
        plt.clf()


if __name__ == '__main__':
    '''
    $ source ../env2/bin/activate
    $ python test.py densenet models/bad16_densenet_bn_static-fbank-0.019326000-sub4.npz hdf5/
    '''

    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Tests a neural network on Bird data using Lasagne.")
        print("Usage: %s [model_type [model_path [hdf5_file_path [csv_file_path]]]]" % sys.argv[0])
        print()
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['filename'] = sys.argv[2]
        if len(sys.argv) > 3:
            hdf5filename = sys.argv[3]
        if len(sys.argv) > 4:
            csvfile = sys.argv[4]


    feature_type='fbank'
    print 'INFO: features = ', feature_type

    subset = 'Test'
    corpus='bad2016test'# corpus de test

    train_corpus='Train_Test_ff1010bird_warblrb10k_public'

    from config import set_options
    options = set_options(feature_type)
    # if corpus == 'bad2016test':
    #     options["TEST_LABELS"] = False

    # Load the dataset
    print("Loading "+ corpus + "...")

    from fuel.datasets.hdf5 import H5PYDataset

    test_set = H5PYDataset(hdf5filename, which_sets=(subset,))

    print("nb samples: %d"% test_set.num_examples)

    from os.path import dirname
    modeldir = dirname(kwargs['filename'])

    main(train_corpus, corpus, test_set, modeldir, options, test_set.num_examples, csvfile, model=kwargs['model'],
         modelfn=kwargs['filename'], loss_type=options["LOSS"])

    outfile=modeldir + '/%s_%s_%s_%s_%s_%s_probs.csv'%(train_corpus, 'Train', corpus, subset, feature_type, kwargs['model'])

