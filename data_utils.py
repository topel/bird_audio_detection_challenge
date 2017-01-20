from create_hdf5_ff1010bird_public import read_target_file
import numpy as np
import shlex, subprocess

def write_meta_output_file(fname, info):
    with open(fname, 'w') as f:
        f.write('itemid,hasbird\n')
        for el in info:
            f.write('%s\n'%el)


def read_csv_filelist(csvfile, hasGT=False):
    passFirstLine=True
    noms = []
    gt = []

    with open(csvfile, 'r') as fh:
        if hasGT:
            for line in fh:
                if passFirstLine:
                    passFirstLine = False
                    continue
                tmp = line.rstrip().split(',')
                noms.append(tmp[0])
                gt.append(tmp[1])
        else:
            for line in fh:
                if passFirstLine:
                    passFirstLine = False
                    continue
                tmp = line.rstrip().split(',')
                noms.append(tmp[0])
    if hasGT:
        return noms, gt
    return noms


def create_hdf5_from_arrays(X, y, subset, h5filename):
    # create an HDF5 file with X, y
    import h5py

    nb_samples, nb_channels, nb_frames, nb_features = X.shape

    f = h5py.File(h5filename, mode='w')
    features = f.create_dataset(
        'features', (nb_samples, nb_channels, nb_frames, nb_features), dtype = 'float32')

    targets = f.create_dataset(
        'targets', (nb_samples, ), dtype = 'uint8')

    features[...] = X
    targets[...] = y

    features.dims[0].label = 'batch'
    features.dims[1].label = 'channel'
    features.dims[2].label = 'width'
    features.dims[3].label = 'height'
    targets.dims[0].label = 'batch'
    # targets.dims[1].label = 'index'

    from fuel.datasets.hdf5 import H5PYDataset

    # split_dict = {
    # 'AugmentTrain': {'features': (0, nb_samples), 'targets': (0, nb_samples)}
    # }
    # print 'INFO: Augmented Train:', (0, nb_samples)

    split_dict = {
    subset: {'features': (0, nb_samples), 'targets': (0, nb_samples)}
    }
    print 'INFO: %s:'%subset, (0, nb_samples)

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

    # train_set = H5PYDataset(h5filename, which_sets=('AugmentedTrain',))
    set = H5PYDataset(h5filename, which_sets=(subset,))
    print set.num_examples

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def merge_two_datasets(subset, hdf5filename1, dataset_name1, noms1, target_dic1, hdf5filename2, dataset_name2, noms2, target_dic2, output, fname, filelist_output, feature_type, withShuffle):

    from fuel.datasets.hdf5 import H5PYDataset

    set1 = H5PYDataset(hdf5filename1, which_sets=(dataset_name1,))
    print set1.num_examples

    set2 = H5PYDataset(hdf5filename2, which_sets=(dataset_name2,))
    print set2.num_examples

    handle1 = set1.open()
    data1 = set1.get_data(handle1, slice(0, set1.num_examples))
    set1.close(handle1)
    X1 = data1[0]
    y1 = data1[1]

    handle2 = set2.open()
    data2 = set2.get_data(handle2, slice(0, set2.num_examples))
    set2.close(handle2)
    X2 = data2[0]
    y2 = data2[1]

    print X1.shape, X2.shape, y1.shape, y2.shape

    X = np.vstack((X1, X2))
    y = np.vstack((y1[:,np.newaxis], y2[:,np.newaxis]))
    y = np.squeeze(y)

    print 'DEBUG:', len(noms1), len(noms2)

    noms = noms1 + noms2
    nb_samples = len(noms)

    print noms1[:5], noms2[:5], noms[:5], noms[len(noms1):len(noms1)+5]

    target_dic = merge_dicts(target_dic1, target_dic2)

    print 'new dataset: ', X.shape, y.shape, nb_samples, len(target_dic.keys())
    
    # if nb_samples != len(target_dic.keys()): raise Exception("Inconsistent number of samples in noms and target_dic")

    if withShuffle:
        np.random.seed(123)
        indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=False)
        print indices[:10]
        Xarray = X[indices]
        yarray = y[indices]
        noms_shuffled = [noms[i] for i in indices]
        # print 0, indices[0], noms_shuffled[0], noms[indices[0]]
        # print X[indices[0]], '\n\n'
        # print Xarray[0]
        noms = list(noms_shuffled)
        del noms_shuffled, X, y
    else:
        indices = range(nb_samples)
        Xarray = X[indices]
        yarray = y[indices]
        del X, y

    indices = range(nb_samples)

    print 'INFO: Xarray:', Xarray.shape, 'yarray:', yarray.shape, len(noms)

    create_hdf5_from_arrays(Xarray, yarray, subset, output)

    if filelist_output is not None:
        from output_utils import save_filelist
        save_filelist(filelist_output, feature_type, indices, noms, target_dic, subset, several_subsets_in_a_single_file=False, train_nb_samples=nb_samples, valid_nb_samples=0, test_nb_samples=0, remove_files=False)

    import cPickle as pickle

    im_mean = np.mean(Xarray, axis=0)
    im_std = np.std(Xarray, axis=0, dtype=np.float64)

    out = open(fname, 'wb')
    pickle.dump({'moyenne': im_mean, 'ecart_type': im_std}, out)
    out.close()

    print 'INFO: written mean / std to PKL file:', fname


def read_remove_file_list(list_file):
    noms = []
    with open(list_file, 'r') as fh:
        for line in fh:
            tmp = line.rstrip().split(',')
            noms.append(tmp[0])
    return noms


def remove_files_from_dataset(hdf5filename1, subset, noms, noms_to_remove, output_):
    from fuel.datasets.hdf5 import H5PYDataset

    set1 = H5PYDataset(hdf5filename1, which_sets=(subset,))
    print 'before:', set1.num_examples

    handle1 = set1.open()
    data1 = set1.get_data(handle1, slice(0, set1.num_examples))
    set1.close(handle1)

    Xarray = []
    yarray = []

    for ind, nom in enumerate(noms):
        if nom in noms_to_remove: continue
        Xarray.append(data1[0][ind])
        yarray.append(data1[1][ind])
    Xarray = np.asarray(Xarray)
    yarray = np.asarray(yarray)

    print 'after:', Xarray.shape

    create_hdf5_from_arrays(Xarray, yarray, output_)

if __name__ == '__main__':
    cavaco = False
    if cavaco:
        # corpus='warblrb10k_public'
        corpus = 'ff1010bird'
        corpusdir='/home/pellegri/corpus/' + corpus
        # corpusdir='/homelocal/corpora/' + corpus
        fbankdir=corpusdir + '/fbank'
        hdf5dir='hdf5'
    else:
        corpus = 'ff1010bird'
        corpusdir='/baie/corpus/BAD2016'
        hdf5dir=corpusdir + '/hdf5'


    feature_type='fbank'
    # feature_type = 'fbank_d_dd'
    # subset='Train'
    # subset='Valid'
    subset='Test'
    withShuffle=True

    # merge subsets from FF and WAR corpus

    corpus1='ff1010bird'
    if feature_type == 'fbank':
        hdf5filename1=hdf5dir + '/%s_melLogSpec56.hdf5'%(corpus1)
    elif feature_type == 'fbank_d_dd':
        hdf5filename1=hdf5dir + '/%s_melLogSpec56deltas.hdf5'%(corpus1)
    filelist1=corpusdir+'/%s/%s_files.csv'%(corpus1, subset)
    noms1 = read_csv_filelist(filelist1)
    target_dic1 = {}
    read_target_file(corpusdir + '/' + corpus1 + '/' + corpus1 + '_metadata.csv', target_dic1, hasGT=True)

    corpus2='warblrb10k_public'
    if feature_type == 'fbank':
        hdf5filename2=hdf5dir + '/%s_melLogSpec56.hdf5'%corpus2
    elif feature_type == 'fbank_d_dd':
        hdf5filename2=hdf5dir + '/%s_melLogSpec56deltas.hdf5'%(corpus2)
    filelist2=corpusdir+'/%s/%s_files.csv'%(corpus2, subset)
    noms2 = read_csv_filelist(filelist2)
    target_dic2 = {}
    read_target_file(corpusdir + '/' + corpus2 + '/' + corpus2 + '_metadata.csv', target_dic2, hasGT=True)

    if feature_type == 'fbank':
        output=hdf5dir + '/%s_%s_%s_melLogSpec56.hdf5'%(subset, corpus1, corpus2)
        mean_fname_output=hdf5dir + '/mean_%s_%s_%s_fbank.pkl'%(subset, corpus1, corpus2)
        filelist_output = corpusdir + '/%s_%s/%s_%s_%s_files.csv'%(corpus1, corpus2, subset, corpus1, corpus2)
    elif feature_type == 'fbank_d_dd':
        output=hdf5dir + '/%s_%s_%s_melLogSpec56deltas.hdf5'%(subset, corpus1, corpus2)
        mean_fname_output=hdf5dir + '/mean_%s_%s_%s_fbankdeltas.pkl'%(subset, corpus1, corpus2)
        filelist_output = None

    print filelist1, filelist2
    merge_two_datasets(subset, hdf5filename1, subset, noms1, target_dic1, hdf5filename2, subset, noms2, target_dic2, output, mean_fname_output, filelist_output, feature_type, withShuffle)

    # merge Train and Test
    corpus1='ff1010bird'
    corpus2='warblrb10k_public'
    subset1 = 'Train'
    if feature_type == 'fbank':
        hdf5filename1=hdf5dir + '/Train_ff1010bird_warblrb10k_public_melLogSpec56.hdf5'
    elif feature_type == 'fbank_d_dd':
        hdf5filename1=hdf5dir + '/Train_ff1010bird_warblrb10k_public_melLogSpec56deltas.hdf5'
    filelist1=corpusdir + '/%s_%s/%s_%s_%s_files.csv'%(corpus1, corpus2, subset1, corpus1, corpus2)
    noms1 = read_csv_filelist(filelist1)
    target_dic1 = {}
    read_target_file(corpusdir + '/' + corpus1 + '/' + corpus1 + '_metadata.csv', target_dic1, hasGT=True)

    subset2='Test'
    if feature_type == 'fbank':
        hdf5filename2=hdf5dir + '/Test_ff1010bird_warblrb10k_public_melLogSpec56.hdf5'
    elif feature_type == 'fbank_d_dd':
        hdf5filename2=hdf5dir + '/Test_ff1010bird_warblrb10k_public_melLogSpec56deltas.hdf5'
    filelist2=corpusdir + '/%s_%s/%s_%s_%s_files.csv'%(corpus1, corpus2, subset2, corpus1, corpus2)
    noms2 = read_csv_filelist(filelist2)
    target_dic2 = {}
    read_target_file(corpusdir + '/' + corpus2 + '/' + corpus2 + '_metadata.csv', target_dic2, hasGT=True)

    if feature_type == 'fbank':
        output=hdf5dir + '/%s_%s_%s_%s_melLogSpec56.hdf5'%(subset1, subset2, corpus1, corpus2)
        mean_fname_output=hdf5dir + '/mean_%s_%s_%s_%s_fbank.pkl'%(subset1, subset2, corpus1, corpus2)
        filelist_output = corpusdir + '/%s_%s/%s_%s_%s_%s_files.csv'%(corpus1, corpus2, subset1, subset2, corpus1, corpus2)
    elif feature_type == 'fbank_d_dd':
        output=hdf5dir + '/%s_%s_%s_%s_melLogSpec56deltas.hdf5'%(subset1, subset2, corpus1, corpus2)
        mean_fname_output=hdf5dir + '/mean_%s_%s_%s_%s_fbankdeltas.pkl'%(subset1, subset2, corpus1, corpus2)
        filelist_output = None

    subset = subset1 + '_' + subset2
    merge_two_datasets(subset, hdf5filename1, subset1, noms1, target_dic1, hdf5filename2, subset2, noms2, target_dic2, output, mean_fname_output, filelist_output, feature_type, withShuffle)
