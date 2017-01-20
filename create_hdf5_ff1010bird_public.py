from scipy.io import loadmat
# import matplotlib.pyplot as plt
import numpy as np
from output_utils import read_target_file, save_filelist

if __name__ == '__main__':
    corpus='ff1010bird'
    # corpusdir='/homelocal/corpora/' + corpus
    corpusdir='/baie/corpus/BAD2016/' # + corpus

    # feature_type='mfcc'
    # feature_type='fbank'
    # feature_type='fbank_d_dd'
    # feature_type='fft'
    # feature_type='slicedfft'
    # feature_type='fp'
    feature_type='fp3'

    augment=False

    if feature_type == 'fbank':
        if augment:
            fbankdir=corpusdir + '/augment_fbank'
            hdf5dir=corpusdir + '/augment_hdf5'
        else:
            fbankdir = corpusdir + '/' + corpus + '/fbank'
            hdf5dir = corpusdir + '/hdf5'
    elif feature_type == 'fbank_d_dd':
        fbankdir=corpusdir + '/' + corpus + '/fbank_delta_deltadelta'
        hdf5dir=corpusdir + '/hdf5'
    elif feature_type == 'fp':
        fbankdir=corpusdir + '/' + corpus + '/fp192x200'
        hdf5dir=corpusdir + '/hdf5'
    elif feature_type == 'fp3':
        fbankdir=corpusdir + '/' + corpus + '/fp132x132x3'
        hdf5dir=corpusdir + '/hdf5'
    elif feature_type == 'fft':
        fbankdir=corpusdir + '/fft'
        hdf5dir=corpusdir + '/hdf5'
    elif feature_type == 'slicedfft':
        fbankdir=corpusdir + '/' + corpus + '/slicedfft'
        hdf5dir=corpusdir + '/hdf5'
    elif feature_type == 'mfcc':
        fbankdir=corpusdir + '/' + corpus + '/mfcc'
        hdf5dir=corpusdir + '/hdf5'

    target_dic = {}
    if augment:
        read_target_file(corpusdir + '/augment_' + corpus + '_metadata.csv', target_dic, hasGT=True)
    else:
        read_target_file(corpusdir + '/' + corpus + '/' + corpus + '_metadata_corrected.csv', target_dic, hasGT=True)

    print 'INFO: nb of files = %d'%(len(target_dic.keys()))

    remove_files = False
    if remove_files:
        list_file = 'models/ff1010bird_warblrb10k_public/best_densenet_fbank_depth19_no_centering/to_remove_ff1010bird_Train.txt'
        with open(list_file, 'r') as fh:
            for line in fh:
                tmp = line.rstrip().split(',')
                id=tmp[0]
                if id in target_dic: del target_dic[id]
        print 'INFO: after removal, nb of files = %d'%(len(target_dic.keys()))

    Xlist = []
    ylist = []
    noms = []

    # nb=0
    nb_positive_class = 0
    nb_negative_class = 0

    print 'loading %s files...'%feature_type
    nb_samples = 0
    if feature_type == 'fbank' or feature_type == 'fbank_d_dd':
        for nom, target in target_dic.iteritems():
            data = loadmat(fbankdir + '/' + nom + '_melLogSpec56.mat')
            data = data['data']
            # plt.imshow(data.T, aspect='auto', origin='lower')
            # plt.show()

            # print nom, target, data.shape

            Xlist.append(data)
            ylist.append(target)
            noms.append(nom)

            if target == '1':
                nb_positive_class += 1
            else:
                nb_negative_class += 1
            nb_samples += 1
            if nb_samples % 200 == 0: print "loaded %d samples"%nb_samples
    elif feature_type == 'fft' or feature_type == 'fp' or feature_type == 'fp3':
        for nom, target in target_dic.iteritems():
            data = np.load(open(fbankdir + '/' + nom + '.npy'))
            # print nom, target, data.shape
            Xlist.append(data)
            ylist.append(target)
            noms.append(nom)

            if target == '1':
                nb_positive_class += 1
            else:
                nb_negative_class += 1
    elif feature_type == 'slicedfft':
        for nom, target in target_dic.iteritems():
            data = np.load(open(fbankdir + '/' + nom + '.npy'))
            Xlist.append(data)
            nb_frames = data.shape[0]
            ylist.extend([np.uint8(target)]*nb_frames)
            noms.extend([nom]*nb_frames)
            nb_samples += nb_frames
            print nom, data.shape, len([target]*nb_frames), len(Xlist), nb_samples
            if target == '1':
                nb_positive_class += nb_frames
            else:
                nb_negative_class += nb_frames
            # if nb_samples>10000:
            #     break
    elif feature_type == 'mfcc':
        for nom, target in target_dic.iteritems():
            # data = loadmat(fbankdir + '/' + nom + '_mfcc13.mat')
            data = loadmat(fbankdir + '/' + nom + '_mfcc56.mat')
            data = data['data']
            # plt.imshow(data.T, aspect='auto', origin='lower')
            # plt.show()

            # print nom, target, data.shape

            Xlist.append(data)
            ylist.append(target)
            noms.append(nom)

            if target == '1':
                nb_positive_class += 1
            else:
                nb_negative_class += 1


    # print noms[:10]

    print 'finished loading'
    print 'creating array...'

    if feature_type == 'slicedfft':
        Xarray = np.vstack(Xlist)
    else:
        Xarray = np.asarray(Xlist, dtype='float32')
    del Xlist
    print Xarray.shape

    if feature_type == 'fbank_d_dd':
        Xarray = np.ndarray.transpose(Xarray, (0, 3, 1, 2))
    elif feature_type != 'fp3':
        Xarray = Xarray[:, np.newaxis, :, :]

    if feature_type == 'slicedfft':
        yarray = np.squeeze(np.vstack(ylist))
    else:
        yarray = np.squeeze(np.asarray(ylist, dtype='uint8'))

    print 'finished creating array...'

    np.random.seed(123)
    shuffled_indices = np.random.choice(range(len(ylist)), size=len(ylist), replace=False)
    del ylist

    Xarray = Xarray[shuffled_indices]
    yarray = yarray[shuffled_indices]

    print 'INFO: Xarray:', type(Xarray), Xarray.shape, 'yarray:', type(yarray), yarray.shape, 'nb_samples:', nb_samples

    nb_samples, nb_channels, nb_frames, nb_features = Xarray.shape

    # sub-corpus division
    proportions = [0.8, 0.05, 0.15]
    assert sum(proportions) == 1.0, 'ERROR: proportions do not sum to 1'

    train_nb_samples=int(np.floor(nb_samples * proportions[0]))
    val_nb_samples=int(np.floor(nb_samples * proportions[1]))
    test_nb_samples=nb_samples - (train_nb_samples + val_nb_samples)

    print 'DEBUG:', train_nb_samples, val_nb_samples, test_nb_samples

    saveFilelist = True
    if saveFilelist:
        save_filelist(corpusdir + '/%s_%s_files.csv'%('Train', feature_type), feature_type, shuffled_indices, noms, target_dic, 'Train', train_nb_samples, val_nb_samples, test_nb_samples, remove_files)
        save_filelist(corpusdir + '/%s_%s_files.csv'%('Valid', feature_type), shuffled_indices, noms, target_dic, 'Valid', train_nb_samples, val_nb_samples, test_nb_samples, remove_files)
        save_filelist(corpusdir + '/%s_%s_files.csv'%('Test', feature_type), shuffled_indices, noms, target_dic, 'Test', train_nb_samples, val_nb_samples, test_nb_samples, remove_files)

    # print train_nb_samples, val_nb_samples, test_nb_samples, nb_samples
    assert train_nb_samples + val_nb_samples + test_nb_samples == nb_samples, 'ERROR: number of subset samples do not sum to the total nb of samples'

    train_nb_negative_samples = np.sum(yarray[0:train_nb_samples]==0)
    train_nb_positive_samples = np.sum(yarray[0:train_nb_samples]==1)
    valid_nb_negative_samples = np.sum(yarray[train_nb_samples:train_nb_samples+val_nb_samples]==0)
    valid_nb_positive_samples = np.sum(yarray[train_nb_samples:train_nb_samples+val_nb_samples]==1)
    test_nb_negative_samples = np.sum(yarray[train_nb_samples+val_nb_samples:]==0)
    test_nb_positive_samples = np.sum(yarray[train_nb_samples+val_nb_samples:]==1)
    print 'INFO: All 0: %d 1: %d --- Train 0: %d 1: %d --- Valid 0: %d 1: %d --- Test 0: %d 1: %d\n'%(
        nb_negative_class, nb_positive_class,
        train_nb_negative_samples, train_nb_positive_samples,
        valid_nb_negative_samples, valid_nb_positive_samples,
        test_nb_negative_samples, test_nb_positive_samples
    )

    import h5py
    if feature_type == 'fbank':
        if remove_files: h5filename=hdf5dir + '/' + corpus + '_melLogSpec56_selected.hdf5'
        else: h5filename=hdf5dir + '/' + corpus + '_melLogSpec56.hdf5'
    elif feature_type == 'fbank_d_dd':
        h5filename=hdf5dir + '/' + corpus + '_melLogSpec56deltas.hdf5'
    elif feature_type == 'fft':
        h5filename=hdf5dir + '/' + corpus + '_fft430x512.hdf5'
    elif feature_type == 'fp':
        h5filename=hdf5dir + '/' + corpus + '_fp192x200.hdf5'
    elif feature_type == 'fp3':
        h5filename=hdf5dir + '/' + corpus + '_fp132x132x3.hdf5'
    elif feature_type == 'slicedfft':
        h5filename=hdf5dir + '/' + corpus + '_fftXx21x512.hdf5'
    elif feature_type == 'mfcc':
        # h5filename=hdf5dir + '/' + corpus + '_mfcc13.hdf5'
        h5filename=hdf5dir + '/' + corpus + '_mfcc56.hdf5'
    f = h5py.File(h5filename, mode='w')

    features = f.create_dataset(
        'features', (nb_samples, nb_channels, nb_frames, nb_features), dtype = 'float32')

    targets = f.create_dataset(
        'targets', (nb_samples, ), dtype = 'uint8')

    features[...] = Xarray
    targets[...] = yarray

    features.dims[0].label = 'batch'
    features.dims[1].label = 'channel'
    features.dims[2].label = 'width'
    features.dims[3].label = 'height'
    targets.dims[0].label = 'batch'
    # targets.dims[1].label = 'index'

    from fuel.datasets.hdf5 import H5PYDataset

    split_dict = {
    'Train': {'features': (0, train_nb_samples), 'targets': (0, train_nb_samples)},
    'Valid': {'features': (train_nb_samples, train_nb_samples + val_nb_samples), 'targets': (train_nb_samples, train_nb_samples + val_nb_samples)},
    'Test': {'features': (train_nb_samples + val_nb_samples, nb_samples), 'targets': (train_nb_samples + val_nb_samples, nb_samples)}
    }

    print 'INFO: Train:', (0, train_nb_samples), 'Valid:', (train_nb_samples, train_nb_samples + val_nb_samples), 'Test:', (train_nb_samples + val_nb_samples, nb_samples)

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

    train_set = H5PYDataset(h5filename, which_sets=('Train',))
    print train_set.num_examples

    valid_set = H5PYDataset(h5filename, which_sets=('Valid',))
    print valid_set.num_examples

    test_set = H5PYDataset(h5filename, which_sets=('Test',))
    print test_set.num_examples
