
import theano
import theano.tensor as T

import time
from model_utils import *
import lasagne

from sklearn.metrics import roc_auc_score, roc_curve
from output_utils import confusion_matrix

from collections import deque

from math import isnan

def main(corpus, train_set, valid_set, train_mean, train_std, useivec, options, model, num_epochs, lr, isNormed, loss_type):


    NB_CHANNELS, NB_FRAMES, NB_FEATURES, NB_CLASSES, BATCH_SIZE, removeMean, divideStd, doAugment, feature_type = options['NB_CHANNELS'], \
                                                                           options['NB_FRAMES'], \
                                                                           options['NB_FEATURES'], \
                                                                           options['NB_CLASSES'], \
                                                                           options['BATCH_SIZE'],\
                                                                           options['CENTER_DATA'], \
                                                                           options['REDUCE_DATA'], \
                                                                           options['AUGMENT'], \
                                                                           options['FEATURE_TYPE']

    print 'OPTIONS: ', options

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    # use batchnorm?
    useBN = True

    # depth = num_block * n + 1 for some n
    network = build_densenet(input_shape=(None, NB_CHANNELS, NB_FRAMES, NB_FEATURES), input_var=input_var, classes=NB_CLASSES,
                   depth=19, first_output=32, growth_rate=15, num_blocks=3,
                   dropout=0, feature_type=feature_type)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)


    if loss_type == 'categorical_crossentropy':
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    elif loss_type == 'binary_hinge':
        prediction = T.extra_ops.squeeze(prediction)
        loss = lasagne.objectives.binary_hinge_loss(prediction, target_var)
    elif loss_type == 'binary_crossentropy':
        prediction = T.extra_ops.squeeze(prediction)
        loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    elif loss_type == 'weighted_binary_crossentropy':
        prediction = T.extra_ops.squeeze(prediction) # test_probas type is col, need a vector to work properly
        if corpus == 'ff1010bird':
            w_pos=1935.0/(1935.0 + 5755.0) # nb of pos examples / nb of examples
        else:
            raise Exception('Please define a weight!')
        loss = weighted_binary_crossentropy(prediction, target_var, w_pos=w_pos)

    loss = loss.mean()

    initial_lr = lr
    # LR_SCHEDULE={
    #     10: 0.1 * initial_lr,
    #     25: 0.01 * initial_lr
    # }

    L2coeff_SCHEDULE={
        0: 0.0
    }

    all_layers = lasagne.layers.get_all_layers(network)
    params = lasagne.layers.get_all_params(network, trainable=True)
    print 'INFO: total number of layers:', len(all_layers)
    print("INFO: number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))

    l_r = theano.shared(lasagne.utils.floatX(lr))

    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=l_r, momentum=0.9)

    test_probas = lasagne.layers.get_output(network, deterministic=True)

    test_loss = lasagne.objectives.categorical_crossentropy(test_probas, target_var)
    # As a bonus, also create an expression for the classification accuracy:
    test_predictions = T.argmax(test_probas, axis=1)
    test_acc = T.mean(T.eq(test_predictions, target_var),
                      dtype=theano.config.floatX)

    test_loss = test_loss.mean()
    test_acc = test_acc.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_probas, test_predictions])

    print 'Finished compiling functions...'
    #
    #

    # A full pass over the validation data:
    val_loss, val_acc, val_batches, val_pred_probs, val_pred, val_gt = test_model(valid_set, train_mean, train_std, removeMean, divideStd, NB_CLASSES, feature_type, val_fn)
    val_roc_auc_score = roc_auc_score(val_gt, val_pred_probs[:,1], average='macro')

    print("Before Training")
    print("  validation loss: {:.6f}\t\t acc: {:.2f} %\t\t auc: {:.2f}\t\t nb_batches: {:.2f}".format(val_loss / val_batches, val_acc / val_batches * 100, 100. * val_roc_auc_score, val_batches))

    LABELS = {'0':0, '1': 1}

    # Finally, launch the training loop.
    print("Starting training...")

    best_valid_accuracy = val_acc / val_batches
    best_valid_auc = val_roc_auc_score
    train_loss_liste = []
    valid_loss_liste = []
    valid_err= []
    valid_auc_neg = []
    test_auc_neg = []
    best_epoch = -1
    best_params = None
    nb_add_params = 0
    # auc_queue = deque([-1, -1, -1])

    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_loss = 0
        train_batches = 0
        # auc_queue.popleft()

        # change learning rate according to schedules
        # if epoch in LR_SCHEDULE:
        #     lr = np.float32(LR_SCHEDULE[epoch])
        #     l_r.set_value(lr)

        print '  current lr: %.12f' % lr

        # # change l2 regul rate according to schedules
        # if epoch in L2coeff_SCHEDULE:
        #     regul = L2coeff_SCHEDULE[epoch]
        #     l2coeff.set_value(regul)
        #     print '  current regul coeff: %.9f' % regul

        start_time = time.time()

        # training
        handle = train_set.open()
        for batch in iterate_minibatches_hdf5(train_set, handle, BATCH_SIZE, feature_type, shuffle=True):
            inputs, targets = batch

            if removeMean:
                # remove mean image
                inputs -= train_mean
                if divideStd:
                    inputs /= train_std

            if doAugment:
                inputs = augment(inputs)

            train_loss += train_fn(inputs, targets)
            train_batches += 1

        train_set.close(handle)
        train_loss_liste.append(train_loss / train_batches)

        if isnan(train_loss): return

        val_loss, val_acc, val_batches, val_pred_probs, val_pred, val_gt = test_model(valid_set, train_mean, train_std, removeMean, divideStd, NB_CLASSES, feature_type, val_fn)

        current_val_acc = val_acc / val_batches
        valid_loss_liste.append(val_loss / val_batches)
        valid_err.append(1.0 - current_val_acc)

        val_roc_auc_score = roc_auc_score(val_gt, val_pred_probs[:, 1], average='macro')

        valid_auc_neg.append(1.0 - val_roc_auc_score)

        ratio = train_loss / val_loss * val_batches / train_batches

        epochDuration = time.time() - start_time
        print("**** Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, epochDuration
        ))
        print("  lr: {:.9f}\t\t train loss: {:.6f}\t\t validation loss: {:.6f}\t\t acc: {:.6f}\t\t auc: {:.6f}\t\t ratio: {:.6f}\t\t ".format(lr,
                                                                                                                        train_loss / train_batches,
                                                                                                                        val_loss / val_batches,
                                                                                                                        current_val_acc * 100,
                                                                                                                        val_roc_auc_score * 100,
                                                                                                                        ratio * 100))

        print('  valid set CM:')
        _, _ = confusion_matrix(LABELS=LABELS, y_test=val_gt, test_pred=val_pred)


        # save model params
        delta_acc = current_val_acc - best_valid_accuracy
        if delta_acc > 0:
            best_valid_accuracy = current_val_acc
            print "  best accuracy update: {:.2f} %".format(best_valid_accuracy * 100)

        delta_auc = val_roc_auc_score - best_valid_auc
        if delta_auc > 0:
            best_valid_auc = val_roc_auc_score
            print "  best auc score update: {:.2f} %".format(best_valid_auc * 100)

        if epoch >=8: # si trois scores consecutifs diminuent, on divise lr par deux
            # if auc_queue[0] > auc_queue[1] > auc_queue[2]:
            lr /= 2.
            l_r.set_value(lasagne.utils.floatX(lr))

        if val_roc_auc_score > 0.98 :
            print ' INFO: adding model params: nb: %d'%(nb_add_params)
            if final_params is None:
                final_params = lasagne.layers.get_all_param_values(network)
            else:
                for i, el in enumerate(final_params):
                    final_params[i] += lasagne.layers.get_all_param_values(network)[i]
            nb_add_params += 1

    if nb_add_params > 0 :
        print ' INFO: nb_add_params =', nb_add_params
        for i, el in enumerate(final_params):
            final_params[i] /= 1. * nb_add_params

    # After training, we compute and print the test error:
    doEval = False
    if doEval:
        test_err, test_acc, test_batches, test_pred_probs, test_pred, test_gt = test_model(test_set, train_mean, train_std, removeMean, divideStd, NB_CLASSES, feature_type, val_fn)
        print("After Training")

        # Compute ROC curve and ROC area class +1
        if NB_CLASSES > 1:
            test_roc_auc_score = roc_auc_score(test_gt, test_pred_probs[:, 1], average='macro')
        else:
            test_roc_auc_score = roc_auc_score(test_gt, test_pred_probs, average='macro')

        print("Test set: loss: {:.6f}\t\t acc: {:.2f} %\t\t auc: {:.2f}".format(test_err / test_batches,
                                                                             test_acc / test_batches * 100,
                                                                             100. * test_roc_auc_score))
        _, _ = confusion_matrix(LABELS=LABELS, y_test=test_gt, test_pred=test_pred)

    doPlot=True
    if doPlot:
        # plot roc curve for test: class 1 only
        if NB_CLASSES > 1:
            fpr, tpr, thr = roc_curve(test_gt, test_pred_probs[:, 1])
        else:
            fpr, tpr, thr = roc_curve(test_gt, test_pred_probs)

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
        plt.savefig('plots/roc.png')
        plt.clf()
        # plt.show()

    # Optionally, you could now dump the network weights to a file like this:
    doSaveModel=True
    if doSaveModel and best_params is not None:
        # if nb_add_params > 0:
        #     for i, el in enumerate(final_params):
        #         final_params[i] *= nb_add_params

        model_dir = 'models/%s/%s_%s'%(corpus, model, feature_type)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_filename = model_dir + '/bad16_'
        best_single_model_filename = model_dir + '/bad16_'

        if model.startswith('mlp'):
            model_filename += 'mlp_'
            best_single_model_filename += 'singleBest_mlp_'
        elif model.startswith('cnn'):
            model_filename += 'cnn_'
            best_single_model_filename += 'singleBest_cnn_'
        elif model == 'residual':
            model_filename += 'resCNN_'
            best_single_model_filename += 'singleBest_resCNN_'
        elif model == 'newresidual':
            model_filename += 'newresCNN_'
            best_single_model_filename += 'singleBest_newresCNN_'
        elif model == 'simpleresidual':
            model_filename += 'sresCNN_'
            best_single_model_filename += 'singleBest_sresCNN_'
        elif model == 'densenet':
            model_filename += 'densenet_'
            best_single_model_filename += 'singleBest_densenet_'
        elif model == 'cam':
            model_filename += 'cam_'
            best_single_model_filename += 'singleBest_cam_'
        else:
            print "ERROR: cannot save model: model name not recognized"


        if useBN:
            model_filename += 'bn_static-%s-%.9f-id%d.npz' % (feature_type, initial_lr, nb_model)
            best_single_model_filename += 'bn_static-%s-%.9f-id%d.npz' % (feature_type, initial_lr, nb_model)
        else:
            model_filename += 'nobn_static-%s.npz'%(feature_type)
            best_single_model_filename += 'nobn_static-%s.npz'%(feature_type)

        np.savez(best_single_model_filename, best_params)
        np.savez(model_filename, final_params)
        print 'INFO: model %s saved!'%(model_filename)
        print 'INFO: model %s saved!'%(best_single_model_filename)


    # plot loss and accuracy
    doPlot=True
    if doPlot:
        train_loss_liste = np.array(train_loss_liste)
        plt.plot(train_loss_liste, label='train loss', color='k')
        plt.legend(loc=2)
        valid_loss_liste = np.array(valid_loss_liste)
        plt.plot(valid_loss_liste, label='valid loss', color='darkgray')
        plt.legend(loc=2)
        plt.ylabel('Categorical Cross Entropy Loss')
        plt.xlabel('Epoch')
        #plt.ylim([0,1.5])
        plt.twinx()
        plt.ylabel('Valid Acc Error and (1-AUC score) (%)')
        plt.grid()
        valid_err = np.array(valid_err)
        plt.plot(valid_err, label='valid error (%)', color='r')
        plt.legend(loc=1)
        plt.plot(valid_auc_neg, label='valid 1-AUC (%)', color='mediumblue')
        plt.legend(loc=1)
        plt.plot(test_auc_neg, label='test 1-AUC (%)', color='lightblue')
        plt.legend(loc=1)
        plt.savefig('plots/log.png')
        plt.clf()
        # plt.show()

if __name__ == '__main__':

    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['model'] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs['num_epochs'] = int(sys.argv[2])

    useZCA = False

    feature_type='fbank'
    # feature_type='fbank_d_dd'
    # feature_type='fft'
    # feature_type='slicedfft'
    # feature_type='mfcc'
    # feature_type='ivec'
    # feature_type='ivecsd'
    # feature_type='fp'
    # feature_type='fp3'

    useivec = False

    print 'INFO: features = ', feature_type

    # corpus='warblrb10k_public'
    corpus = 'ff1010bird'
    # corpus='ff1010bird' + '_warblrb10k_public'
    test_corpus='ff1010bird'
    corpusdir='/baie/corpus/BAD2016'
    hdf5dir=corpusdir + '/hdf5'
    mean_train_file = hdf5dir + '/mean_Train_' + corpus + '_' + feature_type + '.pkl'

    if feature_type=='fbank':
        hdf5filename=hdf5dir + '/%s_melLogSpec56.hdf5'%corpus
        mean_train_file = hdf5dir + '/mean_Train_' + corpus + '_' + feature_type + '.pkl'
        AugmentedTrain_hdf5filename=hdf5dir + '/AugmentedTrain_' + corpus + '_melLogSpec58.hdf5'
        hdf5ZCAfilename = hdf5dir + '/' + corpus + '_melLogSpec56_ZCA_gcn50.hdf5'
        test_valid_hdf5filename=hdf5dir + '/' + test_corpus + '_melLogSpec56.hdf5'
    elif feature_type=='ivec':
        hdf5filename=hdf5dir + '/' + corpus + '_ivectors.hdf5'
        test_valid_hdf5filename=None
        mean_train_file=None
    elif feature_type=='fp':
        hdf5filename=hdf5dir + '/' + corpus + '_fp192x200.hdf5'
        test_valid_hdf5filename=hdf5dir + '/' + test_corpus + '_fp192x200.hdf5'
        mean_train_file=None
    elif feature_type=='fp3':
        hdf5filename=hdf5dir + '/' + corpus + '_fp132x132x3.hdf5'
        test_valid_hdf5filename=hdf5dir + '/' + test_corpus + '_fp132x132x3.hdf5'
        mean_train_file=None
    elif feature_type=='ivecsd':
        hdf5filename=hdf5dir + '/' + corpus + '_ivectors_sddeltas.hdf5'
        mean_file=None
    elif feature_type=='fbank_d_dd':
        hdf5filename=hdf5dir + '/' + corpus + '_melLogSpec56deltas.hdf5'
        test_valid_hdf5filename=hdf5dir + '/' + test_corpus + '_melLogSpec56deltas.hdf5'
        # mean_train_file=None
    elif feature_type=='fft':
        hdf5filename=hdf5dir + '/' + corpus + '_fft430x512.hdf5'
    elif feature_type=='slicedfft':
        hdf5filename=hdf5dir + '/' + corpus + '_fftXx21x512.hdf5'
    elif feature_type=='mfcc':
        # hdf5filename=hdf5dir + '/' + corpus + '_mfcc13.hdf5'
        hdf5filename=hdf5dir + '/' + corpus + '_mfcc56.hdf5'


    from config import set_options
    options = set_options(feature_type)

    # Load the dataset
    print "Loading "+ corpus + "..."

    from fuel.datasets.hdf5 import H5PYDataset

    if useZCA:
        train_set = H5PYDataset(hdf5ZCAfilename, which_sets=('Train',))
        valid_set = H5PYDataset(hdf5ZCAfilename, which_sets=('Valid',))
        test_set = H5PYDataset(hdf5ZCAfilename, which_sets=('Test',))
    else:
        train_set = H5PYDataset(hdf5filename, which_sets=('Train',))
        valid_set = H5PYDataset(hdf5filename, which_sets=('Valid',))

    print train_set.num_examples, valid_set.num_examples

    # load mean image
    if mean_train_file is not None:
        import cPickle as pickle
        h = open(mean_train_file, 'r')
        train_stats = pickle.load(h)
        train_mean = train_stats['moyenne']
        train_std = train_stats['ecart_type']
        h.close()
    else:
        train_mean = None
        train_std = None

    # OK for densenets with depth=19, first_output=32, growth_rate=15, num_blocks=3:
    lr=0.019326

    main(corpus, train_set, valid_set, train_mean, train_std, useivec, options, model=kwargs['model'], num_epochs=kwargs['num_epochs'], lr=lr,
        isNormed=False, loss_type=options["LOSS"])
