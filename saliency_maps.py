import theano
import theano.tensor as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

import time

from model_utils import *
from output_utils import confusion_matrix, save_predictions

def compile_saliency_function(input_layer, output_layer_1):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    # inp = net['input'].input_var
    # outp = lasagne.layers.get_output(net['fc8'], deterministic=True)

    inp = input_layer.input_var
    outp = lasagne.layers.get_output(output_layer_1, deterministic=True)

    max_outp = T.max(outp, axis=1)
    saliency = theano.grad(max_outp.sum(), wrt=inp)
    max_class = T.argmax(outp, axis=1)
    return theano.function([inp], [saliency, max_class])

def show_images(img_original, saliency, max_class, title, nom):
    classes = ['no-bird', 'bird']
    # get out the first map and class from the mini-batch
    # saliency = saliency[0]
    max_class = max_class[0]

    print 'saliency shape:', saliency.shape, 'max_classe:', max_class
    # # convert saliency from BGR to RGB, and from c01 to 01c
    # saliency = saliency[::-1].transpose(1, 2, 0)

    # plot the original image and the three saliency map variants
    plt.figure(figsize=(10, 10), facecolor='w')
    plt.suptitle("Class: " + classes[max_class] + ". Saliency: " + title)
    plt.subplot(2, 2, 1)
    plt.title('input')
    # plt.imshow(img_original)
    plt.imshow(img_original.T, aspect='auto', origin='lower')
    plt.subplot(2, 2, 2)
    plt.title('abs. saliency')
    # if saliency is (w, h, 3) with RGB
    # plt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    plt.imshow(np.abs(saliency).T, aspect='auto', origin='lower', cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('pos. saliency')
    plt.imshow((np.maximum(0, saliency) / saliency.max()).T, aspect='auto', origin='lower')
    plt.subplot(2, 2, 4)
    plt.title('neg. saliency')
    plt.imshow((np.maximum(0, -saliency) / -saliency.min()).T, aspect='auto', origin='lower')
    # plt.show()
    plt.savefig('maps/saliency_%s.png'%nom)
    print 'maps.png saved!'


class ModifiedBackprop(object):

    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self.ops = {}  # memoizes an OpFromGraph instance per tensor type

    def __call__(self, x):
        # OpFromGraph is oblique to Theano optimizations, so we need to move
        # things to GPU ourselves if needed.
        if theano.sandbox.cuda.cuda_enabled:
            maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            maybe_to_gpu = lambda x: x
        # We move the input to GPU if needed.
        x = maybe_to_gpu(x)
        # We note the tensor type of the input variable to the nonlinearity
        # (mainly dimensionality and dtype); we need to create a fitting Op.
        tensor_type = x.type
        # If we did not create a suitable Op yet, this is the time to do so.
        if tensor_type not in self.ops:
            # For the graph, we create an input variable of the correct type:
            inp = tensor_type()
            # We pass it through the nonlinearity (and move to GPU if needed).
            outp = maybe_to_gpu(self.nonlinearity(inp))
            # Then we fix the forward expression...
            op = theano.OpFromGraph([inp], [outp])
            # ...and replace the gradient with our own (defined in a subclass).
            op.grad = self.grad
            # Finally, we memoize the new Op
            self.ops[tensor_type] = op
        # And apply the memoized Op to the input we got.
        return self.ops[tensor_type](x)

class GuidedBackprop(ModifiedBackprop):
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        dtype = inp.dtype
        return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)


def load_a_single_sample(nom, fbankdir):
    data = loadmat(fbankdir + nom + '_melLogSpec56.mat')
    return np.asarray(data['data'][np.newaxis,:,:], dtype='float32')

def run_saliency_maps(network, input_layer, output_layer_1, data, data_filenames):
    # compile the saliency map function, and compute and display the saliency maps
    # img: a single input sample

    print("Compiling saliency_fn...")
    saliency_fn = compile_saliency_function(input_layer, output_layer_1)

    print("Applying saliency_fn...")
    for i in range(data.shape[0]):
        img = data[i,:,:,:]
        img = img[np.newaxis,:,:,:]
        nom = data_filenames[i]
        saliency, max_class = saliency_fn(img)
        print i, img.shape, nom, saliency.shape
        print 'maps/saliency_%s.npz'%nom
        np.savez('maps/saliency_id%d.npz'%i, saliency)
        show_images(img[0,0,:,:], saliency[0,0,:,:], max_class, "guided backprop", nom)

def load_model(modeldir, options, model, modelfn, loss_type):

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

    input_var = T.tensor4('inputs')

    # utiliser des int8 ne marche pas:
    # target_var = T.bvector('targets')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    # use batchnorm?
    network, input_layer, output_layer_1 = build_densenet(input_shape=(None, NB_CHANNELS, NB_FRAMES, NB_FEATURES), input_var=input_var, classes=NB_CLASSES,
               depth=19, first_output=32, growth_rate=15, num_blocks=3,
               dropout=0, feature_type=feature_type)

    print("Loading model...")

    with np.load(modelfn) as f:
        single_array = [f['arr_%d' % i] for i in range(len(f.files))]
        param_values = [el for el in single_array[0]]
    lasagne.layers.set_all_param_values(network, param_values)

    print 'INFO: total number of layers:', len(lasagne.layers.get_all_layers(network))
    print("INFO: number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))

    # replace all the nonlinearities of the network:
    relu = lasagne.nonlinearities.rectify
    relu_layers = [layer for layer in lasagne.layers.get_all_layers(network)
                   if getattr(layer, 'nonlinearity', None) is relu]
    modded_relu = GuidedBackprop(relu)  # important: only instantiate this once!
    for layer in relu_layers:
        layer.nonlinearity = modded_relu

    return network, input_layer, output_layer_1


if __name__ == '__main__':

    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['model'] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs['filename'] = sys.argv[2]
    if len(sys.argv) > 3:
        feature_type = sys.argv[3]
    # if len(sys.argv) > 4:
    #     kwargs['loss_type'] = sys.argv[4]


    useAugmentedTrain=False
    useZCA=False
    cavaco = False

    if len(sys.argv) < 4:
        feature_type='fbank'
        # feature_type='fbank_d_dd'
        # feature_type='fft'
        # feature_type='mfcc'
        # feature_type='slicedfft'
        # feature_type='ivec'
        # feature_type='ivecsd'
        # feature_type='fp'

    print 'INFO: features = ', feature_type

    subset = 'Test'
    # subset='Train'
    # subset='Valid'

    # corpus = 'ff1010bird' # corpus de test
    # corpus='warblrb10k_public'# corpus de test
    corpus='bad2016test'# corpus de test
    # corpus='ff1010bird_warblrb10k_public'# corpus de test

    if corpus == 'bad2016test' and feature_type != 'fbank' and feature_type != 'fbank_d_dd':
        raise Exception("ERROR: with bad2016test, only fbank available fo now!")

    corpusdir='/baie/corpus/BAD2016/' + corpus

    from config import set_options
    options = set_options(feature_type)
    # if corpus == 'bad2016test':
    #     options["TEST_LABELS"] = False


    from os.path import dirname

    modeldir = dirname(kwargs['filename'])

    network, input_layer, output_layer_1 = load_model(modeldir, options, model=kwargs['model'],
         modelfn=kwargs['filename'], loss_type=options["LOSS"])

    # id = '0056c188-b8a5-46d7-ab1e'
    # data = load_a_single_sample(id, fbankdir)
    # print data.shape

    fbankdir=corpusdir + '/fbank/'
    # nb_input_files = 20
    # cpt = 0
    # data_filenames = []
    # passFirstLine=True
    # with open(corpusdir + '/badch_testset_blankresults.csv', 'r') as fh:
    #     for line in fh:
    #         if passFirstLine:
    #             passFirstLine = False
    #             continue
    #         tmp = line.rstrip('\n').split(',')
    #         data_filenames.append(tmp[0])
    #         cpt += 1
    #         if cpt == nb_input_files:
    #             break

    data_filenames = ['a235ab95-9878-437b-8ed4', '375bf073-e669-46b9-b6cf'] # a235ab95-9878-437b-8ed4,0.29722 hidden bird --- 375bf073-e669-46b9-b6cf,0.317429, fake bird
    data = []
    for id in data_filenames:
        data.append(load_a_single_sample(id, fbankdir))
    data = np.asarray(data, dtype='float32')

    print 'data shape:', data.shape
    run_saliency_maps(network, input_layer, output_layer_1, data, data_filenames)
