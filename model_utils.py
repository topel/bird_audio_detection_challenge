import lasagne
import numpy as np
from random import randint

import sys
sys.setrecursionlimit(10000)

def iterate_minibatches_hdf5(dataset, handle, batchsize, feature_type, shuffle=False):

    # inputs and targets should both be 'theano.tensor.sharedvar.TensorSharedVariable' objects
    # with T.cast(y_train, 'int32') when loading the data, it does not work: no get_value() method for y_train, WHY?

    nb_samples = dataset.num_examples
    # assert  nb_samples == targets.get_value(borrow=True).shape[0]
    if shuffle:
        indices = np.arange(nb_samples)
        np.random.shuffle(indices)

    for start_idx in range(0, nb_samples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
            yield dataset.get_data(handle, request=excerpt.tolist())[0], np.int32(
                dataset.get_data(handle, request=excerpt.tolist())[1])
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            yield dataset.get_data(handle, request=excerpt)[0], np.int32(dataset.get_data(handle, request=excerpt)[1])


def augment(batch):
    '''Augmentation - used right before each batch is passed to the network '''
    result = []
    nb_samples, d, w, h = batch.shape

    # pad array
    npad = ((0,0), (0,0), (2,2), (2,2))
    batch = np.pad(batch, pad_width=npad, mode='constant', constant_values=0)


    for i in range(nb_samples):
        x = batch[i]

        # Horizontal flip
        if randint(0, 1) == 1:
          x = x[:,:-1,:]

        x_offset = randint(0, 2)
        y_offset = randint(0, 2)
        result.append(x[:,x_offset : w + x_offset, y_offset : h + y_offset])

    result = np.asarray(result, dtype=np.float32)

    return result


def test_model(dataset, train_mean, train_std, center, divideStd, NB_CLASSES, feature_type, val_fn):
    err_tot = 0
    acc_tot = 0
    batches = 0
    batchsize = 10

    handle = dataset.open()
    nb_samples = dataset.num_examples

    if feature_type == 'fft':
        # slicing_factor = 43
        slicing_factor = 1
    else:
        slicing_factor = 1

    pred = np.zeros(nb_samples * slicing_factor, dtype=np.int32)
    pred_probs = np.zeros((nb_samples * slicing_factor, NB_CLASSES), dtype=np.float32)
    gt_labels = np.zeros(nb_samples * slicing_factor, dtype=np.int32)

    # print 'DEBUG: test_model, taille pred:', pred.shape

    nbs = 0
    for batch in iterate_minibatches_hdf5(dataset, handle, batchsize, feature_type, shuffle=False):
        inputs, targets = batch

        if center:
            inputs -= train_mean
            if divideStd:
                inputs /= train_std

        err, acc, probs, preds = val_fn(inputs, targets)
        err_tot += err
        acc_tot += acc

        pred_probs[batches*batchsize * slicing_factor : (batches+1)*batchsize * slicing_factor] = probs
        pred[batches * batchsize * slicing_factor: (batches + 1) * batchsize * slicing_factor] = preds

        gt_labels[batches*batchsize * slicing_factor: (batches+1)*batchsize * slicing_factor] = targets

        nbs += preds.shape[0]
        batches += 1

    # predictions on remaining test samples:
    remaining_indices=range(batches*batchsize, nb_samples)

    if len(remaining_indices) > 0 :
        inputs, targets = dataset.get_data(handle, request=remaining_indices)[0], np.int32(dataset.get_data(handle, request=remaining_indices)[1])

        weight = 1. * targets.shape[0] / batchsize

        if center:
            # remove mean image
            inputs -= train_mean
            if divideStd:
                inputs /= train_std

        # if feature_type == 'fft':
        #     inputs, targets = decoupeBatch(inputs, targets, slicing_factor)

        err, acc, probs, preds = val_fn(inputs, targets)

        err_tot += err * weight
        acc_tot += acc * weight

        pred_probs[batches*batchsize* slicing_factor:] = probs
        pred[batches * batchsize* slicing_factor:] = preds

        gt_labels[batches*batchsize * slicing_factor:] = targets
        batches += 1. * weight

    # close handle
    dataset.close(handle)

    return err_tot, acc_tot, batches, pred_probs, pred, gt_labels

def build_cnn(input_var=None, nbChannels=1, contextSize=200, featureSize = 56, nb_classes = 2, useBN=True, feature_type='fbank'):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    l_in = lasagne.layers.InputLayer(shape=(None, nbChannels, contextSize, featureSize),
                                        input_var=input_var)


    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 3x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    print 'INFO: use BN = ', useBN
    print 'INFO: input layer: ', l_in.output_shape

    if feature_type == 'fbank':
        first_filter_size = 3
        first_nb_filters = 32
        pool_size = 2
        second_nb_filters = 107
        second_filter_size = 3
        third_nb_filters = 182
        third_filter_size = 3
        fourth_nb_filters = 257
        fourth_filter_size = 3
        pad='same'
    elif feature_type == 'fft':
        first_filter_size = (5, featureSize)
        first_nb_filters = 128
        second_nb_filters = 256
        second_filter_size = (5, 1)
        second_pool_size = (1, 1)
        pad='valid'
    elif feature_type == 'slicedfft':
        # first_filter_size = (2, 401)
        # first_nb_filters = 32
        # second_nb_filters = 256
        # second_filter_size = 2
        # second_pool_size = 2
        # pad='valid'
        first_filter_size = (2, 3)
        first_nb_filters = 32
        second_nb_filters = 64
        second_filter_size = (3,4)
        second_pool_size = 2
        pad='valid'

    if useBN:
        l_conv1 = lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                l_in, num_filters=first_nb_filters, filter_size=first_filter_size, pad=pad, flip_filters=False,
                nonlinearity=None,
                W=lasagne.init.GlorotUniform())
        )
    else:
        l_conv1 = lasagne.layers.Conv2DLayer(
                l_in, num_filters=first_nb_filters, filter_size=first_filter_size, pad=pad,
                nonlinearity=None,
                W=lasagne.init.GlorotUniform()
        )

    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    print('INFO: first conv layer: ', l_conv1.output_shape)

    if useBN:
        l_conv2 = lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                l_conv1, num_filters=second_nb_filters, filter_size=second_filter_size, pad=pad, stride=1, flip_filters=False,
                # l_mp1, num_filters=64, filter_size=(2, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        )
    else:
        l_conv2 = lasagne.layers.Conv2DLayer(
                l_conv1, num_filters=second_nb_filters, filter_size=second_filter_size, pad=pad, stride=1, flip_filters=False,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform()
        )

    # Max-pooling layer of factor 2 in both dimensions:
    l_mp2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=pool_size)
    print('INFO: second conv/MP layer: ', l_conv2.output_shape, l_mp2.output_shape)

    if useBN:
        l_conv3 = lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                l_mp2, num_filters=third_nb_filters, filter_size=third_filter_size, pad=pad, stride=1, flip_filters=False,
                # l_mp1, num_filters=64, filter_size=(2, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        )
    else:
        l_conv3 = lasagne.layers.Conv2DLayer(
                l_mp2, num_filters=third_nb_filters, filter_size=third_filter_size, pad=pad, stride=1, flip_filters=False,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform()
        )

    # Max-pooling layer of factor 2 in both dimensions:
    l_mp3 = lasagne.layers.MaxPool2DLayer(l_conv3, pool_size=pool_size)

    print('INFO: third conv/MP layer: ', l_conv3.output_shape, l_mp3.output_shape)

    if useBN:
        l_conv4 = lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                l_mp3, num_filters=fourth_nb_filters, filter_size=fourth_filter_size, pad=pad, stride=1, flip_filters=False,
                # l_mp1, num_filters=64, filter_size=(2, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        )
    else:
        l_conv4 = lasagne.layers.Conv2DLayer(
                l_mp3, num_filters=fourth_nb_filters, filter_size=fourth_filter_size, pad=pad, stride=1, flip_filters=False,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform()
        )

    print('INFO: fourth conv layer: ', l_conv4.output_shape)

    network = lasagne.layers.GlobalPoolLayer(l_conv4, name='post_pool')
    print('INFO: post Global pool layer: ', network.output_shape)

    if nb_classes == 1:
        network = lasagne.layers.DenseLayer(network, nb_classes, nonlinearity=lasagne.nonlinearities.sigmoid,
                             W=lasagne.init.HeNormal(gain=1), name='output')
    else:
        network = lasagne.layers.DenseLayer(network, nb_classes, nonlinearity=lasagne.nonlinearities.softmax,
                         W=lasagne.init.HeNormal(gain=1), name='output')

    print('INFO: output layer: ', network.output_shape)

    # return network, l_conv1, l_mp1, l_conv2, l_mp2, l_dense1, l_dense2, l_dense3
    # return network, l_conv1, l_mp1, l_conv2, l_mp2, l_dense1
    return network


def build_densenet(input_shape=(None, 1, 200, 56), input_var=None, classes=2,
                   depth=40, first_output=16, growth_rate=12, num_blocks=3,
                   dropout=0, feature_type='fbank'):
    """
    Creates a DenseNet model in Lasagne.
    Parameters
    ----------
    input_shape : tuple
        The shape of the input layer, as ``(batchsize, channels, rows, cols)``.
        Any entry except ``channels`` can be ``None`` to indicate free size.
    input_var : Theano expression or None
        Symbolic input variable. Will be created automatically if not given.
    classes : int
        The number of classes of the softmax output.
    depth : int
        Depth of the network. Must be ``num_blocks * n + 1`` for some ``n``.
        (Parameterizing by depth rather than n makes it easier to follow the
        paper.)
    first_output : int
        Number of channels of initial convolution before entering the first
        dense block, should be of comparable size to `growth_rate`.
    growth_rate : int
        Number of feature maps added per layer.
    num_blocks : int
        Number of dense blocks (defaults to 3, as in the original paper).
    dropout : float
        The dropout rate. Set to zero (the default) to disable dropout.
    batchsize : int or None
        The batch size to build the model for, or ``None`` (the default) to
        allow any batch size.
    inputsize : int, tuple of int or None
    Returns
    -------
    network : Layer instance
        Lasagne Layer instance for the output layer.
    References
    ----------
    .. [1] Gao Huang et al. (2016):
           Densely Connected Convolutional Networks.
           https://arxiv.org/abs/1608.06993
    """
    if (depth - 1) % num_blocks != 0:
        raise ValueError("depth must be num_blocks * n + 1 for some n")

    # input and initial convolution
    network = lasagne.layers.InputLayer(input_shape, input_var, name='input')
    print('INFO: input layer: ', network.output_shape)

    if feature_type == 'fbank' or feature_type == 'fbank_d_dd' or feature_type == 'fp' or feature_type == 'fp3':
        first_filter_size = 3
        filter_size = 1
        dense_block_filter_size = 3
        pool_size = 2
        pad='same'
    elif feature_type == 'fft':
        first_filter_size = (5, input_shape[3])
        filter_size = (5, 1)
        dense_block_filter_size = (5, 1)
        pool_size = (1, 1)
        pad='valid'
    elif feature_type == 'slicedfft':
        first_filter_size = 3
        filter_size = 1
        dense_block_filter_size = 3
        pool_size = 2
        pad='valid'
    elif feature_type == 'mfcc':
        first_filter_size = 3
        filter_size = 1
        dense_block_filter_size = 3
        pool_size = 2
        pad='same'

    # network = lasagne.layers.Conv2DLayer(network, 8, (7, 25), pad=pad,
    #                       W=lasagne.init.HeNormal(gain='relu'),
    #                       b=None, nonlinearity=None, name='prepre_conv')
    # print('INFO: zeroth conv layer: ', network.output_shape)


    network = lasagne.layers.Conv2DLayer(network, first_output, first_filter_size, pad=pad,
                          W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None, name='pre_conv')
    print('INFO: first conv layer: ', network.output_shape)
    # note: The authors' implementation does *not* have a dropout after the
    #       initial convolution. This was missing in the paper, but important.
    # if dropout:
    #     network = DropoutLayer(network, dropout)
    # dense blocks with transitions in between
    n = (depth - 1) // num_blocks
    for b in range(num_blocks):
        network = dense_block(network, n - 1, growth_rate, dense_block_filter_size, dropout,
                              name_prefix='block%d' % (b + 1))
        print('INFO: dense block %d: '%b, network.output_shape)
        if b < num_blocks - 1:
            network = transition(network, dropout, filter_size, pool_size,
                                 name_prefix='block%d_trs' % (b + 1))
            print('INFO: transition %d: '%b, network.output_shape)
    # post processing until prediction
    network = lasagne.layers.BatchNormLayer(network, name='post_bn')
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify,
                                name='post_relu')
    network = lasagne.layers.GlobalPoolLayer(network, name='post_pool')
    print('INFO: post Global pool layer: ', network.output_shape)
    if classes == 1:
        network = lasagne.layers.DenseLayer(network, classes, nonlinearity=lasagne.nonlinearities.sigmoid,
                             W=lasagne.init.HeNormal(gain=1), name='output')
    else:
        network = lasagne.layers.DenseLayer(network, classes, nonlinearity=lasagne.nonlinearities.softmax,
                         W=lasagne.init.HeNormal(gain=1), name='output')

    print('INFO: output layer: ', network.output_shape)

    return network

def build_densenet_with_two_inputs(input_shape=(None, 1, 200, 56), input_var=None, input_var2=None, classes=2,
                   depth=40, first_output=16, growth_rate=12, num_blocks=3,
                   dropout=0, feature_type='fbank'):
    """
    Creates a DenseNet model in Lasagne.
    Parameters
    ----------
    input_shape : tuple
        The shape of the input layer, as ``(batchsize, channels, rows, cols)``.
        Any entry except ``channels`` can be ``None`` to indicate free size.
    input_var : Theano expression or None
        Symbolic input variable. Will be created automatically if not given.
    classes : int
        The number of classes of the softmax output.
    depth : int
        Depth of the network. Must be ``num_blocks * n + 1`` for some ``n``.
        (Parameterizing by depth rather than n makes it easier to follow the
        paper.)
    first_output : int
        Number of channels of initial convolution before entering the first
        dense block, should be of comparable size to `growth_rate`.
    growth_rate : int
        Number of feature maps added per layer.
    num_blocks : int
        Number of dense blocks (defaults to 3, as in the original paper).
    dropout : float
        The dropout rate. Set to zero (the default) to disable dropout.
    batchsize : int or None
        The batch size to build the model for, or ``None`` (the default) to
        allow any batch size.
    inputsize : int, tuple of int or None
    Returns
    -------
    network : Layer instance
        Lasagne Layer instance for the output layer.
    References
    ----------
    .. [1] Gao Huang et al. (2016):
           Densely Connected Convolutional Networks.
           https://arxiv.org/abs/1608.06993
    """
    if (depth - 1) % num_blocks != 0:
        raise ValueError("depth must be num_blocks * n + 1 for some n")

    # input and initial convolution
    network = lasagne.layers.InputLayer(input_shape, input_var, name='input')
    print('INFO: input layer: ', network.output_shape)

    useBN = True
    useDropout=False

    l_in_ivec = lasagne.layers.InputLayer(shape=(None, 1, 600), input_var=input_var2)

    n_units1=1000
    # Convolutional layer with 32 kernels of size 3x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    print 'INFO: use BN = ', useBN
    print 'INFO: input layer i-vectors: ', l_in_ivec.output_shape

    if useBN:
        if useDropout:
            p_dropout = 0.05
            l_dense_ivec = lasagne.layers.batch_norm(
                    lasagne.layers.DenseLayer(
                    lasagne.layers.dropout(l_in_ivec, p=p_dropout), num_units=n_units1,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform()))
        else:
            l_dense_ivec = lasagne.layers.batch_norm(
                    lasagne.layers.DenseLayer(
                    l_in_ivec, num_units=n_units1,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform()))
    else:
        if useDropout:
            p_dropout = 0.05
            l_dense_ivec = lasagne.layers.DenseLayer(
                    lasagne.layers.dropout(l_in_ivec, p=p_dropout), num_units=n_units1,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform())
        else:
            l_dense_ivec = lasagne.layers.DenseLayer(
                    l_in_ivec, num_units=n_units1,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform())
    print 'INFO: ivector dense layer: ', l_dense_ivec.output_shape


    if feature_type == 'fbank' or feature_type == 'fbank_d_dd':
        first_filter_size = 3
        filter_size = 1
        dense_block_filter_size = 3
        pool_size = 2
        pad='same'
    elif feature_type == 'fft':
        first_filter_size = (5, input_shape[3])
        filter_size = (5, 1)
        dense_block_filter_size = (5, 1)
        pool_size = (1, 1)
        pad='valid'
    elif feature_type == 'slicedfft':
        first_filter_size = 3
        filter_size = 1
        dense_block_filter_size = 3
        pool_size = 2
        pad='valid'
    elif feature_type == 'mfcc':
        first_filter_size = 3
        filter_size = 1
        dense_block_filter_size = 3
        pool_size = 2
        pad='same'

    network = lasagne.layers.Conv2DLayer(network, first_output, first_filter_size, pad=pad,
                          W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None, name='pre_conv')
    print('INFO: first conv layer: ', network.output_shape)
    # note: The authors' implementation does *not* have a dropout after the
    #       initial convolution. This was missing in the paper, but important.
    # if dropout:
    #     network = DropoutLayer(network, dropout)
    # dense blocks with transitions in between
    n = (depth - 1) // num_blocks
    for b in range(num_blocks):
        network = dense_block(network, n - 1, growth_rate, dense_block_filter_size, dropout,
                              name_prefix='block%d' % (b + 1))
        print('INFO: dense block %d: '%b, network.output_shape)
        if b < num_blocks - 1:
            network = transition(network, dropout, filter_size, pool_size,
                                 name_prefix='block%d_trs' % (b + 1))
            print('INFO: transition %d: '%b, network.output_shape)
    # post processing until prediction
    network = lasagne.layers.BatchNormLayer(network, name='post_bn')
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify,
                                name='post_relu')
    network = lasagne.layers.GlobalPoolLayer(network, name='post_pool')
    print('INFO: post Global pool layer: ', network.output_shape)

    if classes == 1:
        network = lasagne.layers.DenseLayer([network, l_dense_ivec], classes, nonlinearity=lasagne.nonlinearities.sigmoid,
                             W=lasagne.init.HeNormal(gain=1), name='output')
    else:
        network = lasagne.layers.ConcatLayer([network, l_dense_ivec], axis=1)
        print('INFO: concat layer: ', network.output_shape)

        network = lasagne.layers.DenseLayer(network, classes, nonlinearity=lasagne.nonlinearities.softmax,
                         W=lasagne.init.HeNormal(gain=1), name='output')

    print('INFO: output layer: ', network.output_shape)

    return network

def dense_block(network, num_layers, growth_rate, dense_block_filter_size, dropout, name_prefix):
    # concatenated 3x3 convolutions
    for n in range(num_layers):
        conv = bn_relu_conv(network, channels=growth_rate,
                            filter_size=dense_block_filter_size, dropout=dropout,
                            name_prefix=name_prefix + '_l%02d' % (n + 1))
        network = lasagne.layers.ConcatLayer([network, conv], axis=1,
                              name=name_prefix + '_l%02d_join' % (n + 1))
    return network


def transition(network, dropout, filter_size, pool_size, name_prefix):
    # a transition 1x1 convolution followed by avg-pooling
    network = bn_relu_conv(network, channels=network.output_shape[1],
                           filter_size=filter_size, dropout=dropout,
                           name_prefix=name_prefix)
    network = lasagne.layers.Pool2DLayer(network, pool_size, mode='average_inc_pad',
                          name=name_prefix + '_pool')
    return network


def bn_relu_conv(network, channels, filter_size, dropout, name_prefix):
    network = lasagne.layers.BatchNormLayer(network, name=name_prefix + '_bn')
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify,
                                name=name_prefix + '_relu')
    network = lasagne.layers.Conv2DLayer(network, channels, filter_size, pad='same',
                          W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None,
                          name=name_prefix + '_conv')
    if dropout:
        network = lasagne.layers.DropoutLayer(network, dropout)
    return network


class DenseNetInit(lasagne.init.Initializer):
    """
    Reproduces the initialization scheme of the authors' Torch implementation.
    At least for the 40-layer networks, lasagne.init.HeNormal works just as
    fine, though. Kept here just in case. If you want to swap in this scheme,
    replace all W= arguments in all the code above with W=DenseNetInit().
    """
    def sample(self, shape):
        import numpy as np
        rng = lasagne.random.get_rng()
        if len(shape) >= 4:
            # convolutions use Gaussians with stddev of sqrt(2/fan_out), see
            # https://github.com/liuzhuang13/DenseNet/blob/cbb6bff/densenet.lua#L85-L86
            # and https://github.com/facebook/fb.resnet.torch/issues/106
            fan_out = shape[0] * np.prod(shape[2:])
            W = rng.normal(0, np.sqrt(2. / fan_out),
                           size=shape)
        elif len(shape) == 2:
            # the dense layer uses Uniform of range sqrt(1/fan_in), see
            # https://github.com/torch/nn/blob/651103f/Linear.lua#L21-L43
            fan_in = shape[0]
            W = rng.uniform(-np.sqrt(1. / fan_in), np.sqrt(1. / fan_in),
                            size=shape)
        return lasagne.utils.floatX(W)
