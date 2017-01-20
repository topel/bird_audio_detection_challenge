import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, precision_recall_fscore_support

def confusion_matrix(LABELS, y_test, test_pred):
    # confusion matrix
    from sklearn.metrics import confusion_matrix
    plotCM = False
    target_names = LABELS.keys()
    cm = confusion_matrix(y_test, test_pred)


    if plotCM:
        import matplotlib.pyplot as plt

        plt.figure()
        plot_confusion_matrix(cm, target_names)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if plotCM:
        plt.figure()
        plot_confusion_matrix(cm_normalized, target_names, title='Normalized confusion matrix')
        plt.show()

    np.set_printoptions(precision=1)
    # print('Confusion matrix')
    print(cm)
    print (cm_normalized * 100.)
    # print('Normalized confusion matrix')
    # print(cm_normalized)

    return cm, cm_normalized

def plot_confusion_matrix(cm, target_names, title='Confusion matrix'):
    import matplotlib.pyplot as plt
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def read_target_file(filename, tdic, hasGT):
    """
        reads CSV files with first line being a header
    such as ff1010bird_metadata.csv
    :param filename: GT file path
    :param tdic: empty dictionary to be filled, keys=file ids, values=GT targets
    :return:
    """
    passFirstLine=True
    with open(filename, 'r') as fh:
        if hasGT:
            for line in fh:
                if passFirstLine:
                    passFirstLine = False
                    continue
                tmp = line.rstrip('\n').split(',')
                tdic[tmp[0]] = tmp[1]
        else:
            for line in fh:
                if passFirstLine:
                    passFirstLine = False
                    continue
                tmp = line.rstrip('\n').split(',')
                tdic[tmp[0]] = -1


def save_filelist(fname, feature_type, shuffled_indices, noms, target_dic, subset, several_subsets_in_a_single_file, train_nb_samples, valid_nb_samples, test_nb_samples, remove_files):
    """
    Saves a csv files with the audio file ids (1st field) and the GT targets (2nd field)
    :param corpusdir:
    :param noms: list with the file ids
    :param target_dic: keys=file ids, values=GT targets
    :param subset: train | valid | test
    :param train_nb_samples:
    :param valid_nb_samples:
    :param test_nb_samples:
    :return:
    """
    import csv
    if several_subsets_in_a_single_file:
        if subset == 'Train':
            start=0
            nb_samples = train_nb_samples
        elif subset == 'Valid':
            start = train_nb_samples
            nb_samples = valid_nb_samples
        elif subset == 'Test':
            start = train_nb_samples + valid_nb_samples
            nb_samples = test_nb_samples
    else:
        start=0
        nb_samples = train_nb_samples

    with open(fname, 'w') as csvfile:
        fieldnames = ['itemid', 'hasbird']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(start, start + nb_samples):
            ind = shuffled_indices[i]
            writer.writerow({'itemid': noms[ind], 'hasbird': target_dic[noms[ind]]})
            # if subset == 'test':
            #     print 'itemid', noms[ind], 'hasbird', target_dic[noms[ind]]
    print "INFO: file lists saved to file: %s"%fname


def save_filelist_sliced_features(corpusdir, feature_type, shuffled_indices, noms, dico_noms, target_dic, subset, train_nb_samples, valid_nb_samples, test_nb_samples,
                                  train_nb_samples_in_files, val_nb_samples_in_files, test_nb_samples_in_files):
    """
    Saves a csv files with the audio file ids (1st field) and the GT targets (2nd field)
    :param corpusdir:
    :param noms: list with the file ids
    :param dico_noms: dict with keys: noms and values: nb of frames
    :param target_dic: keys=file ids, values=GT targets
    :param subset: train | valid | test
    :param train_nb_samples:
    :param valid_nb_samples:
    :param test_nb_samples:
    :return:
    """
    import csv
    if subset == 'Train':
        start=0
        nb_samples_in_files = train_nb_samples_in_files
        nb_samples = train_nb_samples
    elif subset == 'Valid':
        start = train_nb_samples_in_files
        nb_samples_in_files = val_nb_samples_in_files
        nb_samples = valid_nb_samples
    elif subset == 'Test':
        start = train_nb_samples_in_files + val_nb_samples_in_files
        nb_samples_in_files = test_nb_samples_in_files
        nb_samples = test_nb_samples

    nb_written_samples = 0
    with open(corpusdir + '/%s_%s_files.csv'%(subset, feature_type), 'w') as csvfile:
        fieldnames = ['itemid', 'hasbird']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(start, start + nb_samples_in_files):
            ind = shuffled_indices[i]
            for j in range(dico_noms[noms[ind]]):
                writer.writerow({'itemid': noms[ind], 'hasbird': target_dic[noms[ind]]})
                nb_written_samples += 1
            # if subset == 'test':
            #     print 'itemid', noms[ind], 'hasbird', target_dic[noms[ind]]

    if nb_written_samples != nb_samples: raise Exception("ERROR in save_filelist_sliced_features: inconsistent number of samples!")
    print "INFO: file lists saved to file:", corpusdir + '/%s_%s_files.csv'%(subset, feature_type)


def save_predictions(gtfilename, loss_type, probs, preds, outfile):
    """
        Save predictions or probabilities to a csv file

    :param gtfilename: metadata csv file with GT
    :param probs: prediction probabilities (numpy array)
    :param loss_type: used to know if probs is a np matrix or a vector
    :param outfile: file path
    :return:
    """

    # 1. get file ids
    liste_fileids = []
    targets = []
    passFirstLine=True
    with open(gtfilename, 'r') as fh:
        for line in fh:
            if passFirstLine:
                passFirstLine = False
                continue
            tmp = line.rstrip().split(',')
            liste_fileids.append(tmp[0])
            targets.append(tmp[1])

    print 'liste_fileids', len(liste_fileids)
    # 2. save preds
    import csv
    with open(outfile, 'w') as csvfile:
        fieldnames = ['itemid', 'hasbird', 'pred', 'gt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if loss_type == 'categorical_crossentropy':
            for i, id in enumerate(liste_fileids):
                writer.writerow({'itemid': id, 'hasbird': probs[i, 1], 'pred': preds[i], 'gt': targets[i]})
        elif loss_type == 'binary_hinge' or loss_type == 'binary_crossentropy' or loss_type == 'weighted_binary_crossentropy':
            for i, id in enumerate(liste_fileids):
                writer.writerow({'itemid': id, 'hasbird': probs[i][0], 'pred': preds[i], 'gt': targets[i]})

    print "INFO: predictions (positive class probas) saved to file:", outfile

def plot_distrib_probas(pred_csv_file, corpus, subcorpus, modeldir):

    # print 'DEBUG:', pred_csv_file
    fp_name = modeldir + '/fp_%s_%s.txt'%(corpus, subcorpus)
    fp_fh = open(fp_name, 'w')
    fn_name = modeldir + '/fn_%s_%s.txt'%(corpus, subcorpus)
    fn_fh = open(fn_name, 'w')

    remove_file_list = modeldir + '/to_remove_%s_%s.txt'%(corpus, subcorpus)
    remove_file_list_fh = open(remove_file_list, 'w')

    dico_preds = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    dico_probs = {'tp': [], 'fp': [], 'tn': [], 'fn': []}
    passFirstLine=True
    i = 0
    with open(pred_csv_file, 'r') as fh:
        for line in fh:
            if passFirstLine:
                passFirstLine = False
                continue
            tmp = line.rstrip().split(',')
            prob = float(tmp[1])
            pred = int(tmp[2])
            gt_pos = int(tmp[3]) == 1
            if gt_pos:
                if pred == 1:
                    dico_preds['tp'] += 1
                    dico_probs['tp'].append(prob)
                else:
                    # print 'fn:', tmp[0], prob, pred, tmp[3]
                    fn_fh.write('%s,%.3f\n'%(tmp[0], prob))
                    remove_file_list_fh.write('%s,%s\n'%(tmp[0], tmp[3]))
                    dico_preds['fn'] += 1
                    dico_probs['fn'].append(prob)
            else:
                if pred == 0:
                    dico_preds['tn'] += 1
                    dico_probs['tn'].append(prob)
                else:
                    # print 'fp:', tmp[0], prob, pred, tmp[3]
                    fp_fh.write('%s,%.3f\n'%(tmp[0], prob))
                    remove_file_list_fh.write('%s,%s\n'%(tmp[0],tmp[3]))
                    dico_preds['fp'] += 1
                    dico_probs['fp'].append(prob)
            i+=1
    fn_fh.close()
    fp_fh.close()
    remove_file_list_fh.close()

    print 'INFO: %s SAVED'%(fp_name)
    print 'INFO: %s SAVED'%(fn_name)
    print 'INFO: %s SAVED'%(remove_file_list)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # from scipy.interpolate import UnivariateSpline

    # for k, v in dico_preds.iteritems():
    #     print k, v
    for k in ['fp', 'fn']:
        v = dico_preds[k]
        print k, v
        p, x = np.histogram(dico_probs[k], bins=10)
        x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
        plt.plot(x, p, label=k)
        # f = UnivariateSpline(x, p, s=10)
        # plt.plot(x, f(x))
    plt.legend(loc=0)
    plt.savefig('plots/distrib.png')


def plot_distrib_probas_noGT(pred_csv_file):

    # print 'DEBUG:', pred_csv_file
    probas = []
    passFirstLine=True
    i = 0
    with open(pred_csv_file, 'r') as fh:
        for line in fh:
            if passFirstLine:
                passFirstLine = False
                continue
            tmp = line.rstrip().split(',')
            probas.append(float(tmp[1]))
            i+=1

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # from scipy.interpolate import UnivariateSpline

    # p, x = np.histogram(probas, bins=10)
    # x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
    # plt.plot(x, p)
    # # f = UnivariateSpline(x, p, s=10)
    # # plt.plot(x, f(x))
    # # plt.legend(loc=0)
    # plt.savefig('plots/distrib.png')

    from os.path import dirname
    outdir = dirname(pred_csv_file)
    plt.figure()
    num_bins=20
    plt.hist(probas, num_bins, alpha=0.5, normed=False)
    # f = UnivariateSpline(x, p, s=10)
    # plt.plot(x, f(x))
    # plt.legend(loc=0)
    plt.savefig(outdir + '/distrib.png')

    plt.figure()
    num_bins=20
    weights = np.ones_like(probas)/float(len(probas))
    plt.hist(probas, num_bins, weights=weights, alpha=0.5)
    # f = UnivariateSpline(x, p, s=10)
    # plt.plot(x, f(x))
    # plt.legend(loc=0)
    plt.savefig(outdir + '/distrib_normed.png')

def read_pred_csv_file_to_arrays(csvfile, hasPred=True):

    passFirstLine=True
    gt = []
    gt_dico_one_per_file = {}
    probs = []
    preds = []
    noms = []

    with open(csvfile, 'r') as fh:
        if hasPred:
            for line in fh:
                if passFirstLine:
                    passFirstLine = False
                    continue
                tmp = line.rstrip().split(',')
                gt.append(int(tmp[3]))
                preds.append(int(tmp[2]))
                probs.append(float(tmp[1]))
                noms.append(tmp[0])
                if tmp[0] not in gt_dico_one_per_file.keys(): gt_dico_one_per_file[tmp[0]] = gt[-1]
        else:
            for line in fh:
                if passFirstLine:
                    passFirstLine = False
                    continue
                tmp = line.rstrip().split(',')
                probs.append(float(tmp[1]))
                noms.append(tmp[0])

    if hasPred:
        return np.asarray(gt), np.asarray(probs), np.asarray(preds), noms, gt_dico_one_per_file
    else:
        return np.asarray(gt), np.asarray(probs), None, noms, None


def predict_sigmoid(a, b, T):
    """Predict new data by linear interpolation.

    Parameters
    ----------
    T : array-like, shape (n_samples,)
        Data to predict from.

    Returns
    -------
    T_ : array, shape (n_samples,)
        The predicted data.
    """
    from sklearn.utils import column_or_1d
    T = column_or_1d(T)
    return 1. / (1. + np.exp(a * T + b))

def binary_predict(probs, threshold = 0.5):
    """Predict the target of new samples. Can be different from the
    prediction of the uncalibrated classifier.

    Parameters
    ----------
    probs : array-like, shape (n_samples, )

    threshold : if above, positive class

    Returns
    -------
    C : array, shape (n_samples,)
        The predicted class.
    """
    return (probs >= threshold) * np.ones(len(probs))

def plot_calibration_curve(classifier_name, pred_csv_file, fig_index):
    """Plot calibration curve for est w/o and with calibration.
        cf http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py
    """

    from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.isotonic import isotonic_regression
    from sklearn.metrics import roc_auc_score, roc_curve, auc

    # # Calibrated with isotonic calibration
    # isotonic = CalibratedClassifierCV(base_estimator=None, cv="prefit", method='isotonic')

    # # Calibrated with sigmoid calibration
    # sigmoid = CalibratedClassifierCV(base_estimator=None, cv="prefit", method='sigmoid')

    # # Logistic regression with no calibration as baseline
    # lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # for name in [classifier_name, classifier_name + ' + Isotonic',  classifier_name + ' + Sigmoid']:
    for name in [classifier_name, classifier_name + ' + Sigmoid']:
    # for name in [classifier_name]:

        y_test, prob_pos, y_pred, _, _ = read_pred_csv_file_to_arrays(pred_csv_file)

        if name == classifier_name + ' + Sigmoid':
            a, b = sigmoid_calibration(prob_pos, y_test, sample_weight=None)
            prob_pos = predict_sigmoid(a, b, prob_pos)
            print a, b
            y_pred = binary_predict(prob_pos, threshold = 0.5)


        if name == classifier_name + ' + Isotonic' :
            prob_pos = isotonic_regression(prob_pos, sample_weight=None, y_min=None, y_max=None,
                        increasing=True)
            y_pred = binary_predict(prob_pos, threshold = 0.5)


        # print prob_pos[:20]
        # # plot roc curve for test: class 1 only
        # fpr, tpr, _ = roc_curve(y_test, prob_pos)
        # lw = 2
        # plt.plot(fpr, tpr, color='darkorange',
        #                        lw=lw, label='ROC curve (area = %0.2f)' %(roc_auc_score(y_test, prob_pos, average='macro')))
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.savefig('plots/roc_%s.png'%(name))
        # plt.clf()

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=1)
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f" % f1_score(y_test, y_pred))
        print("\tROC: %1.3f\n" % roc_auc_score(y_test, prob_pos, average='macro'))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig('plots/calibration.png')
    plt.clf()

def acc_f1_roc(gt, prob, pred):

    acc = accuracy_score(gt, pred)*100.
    acc_not_normed = accuracy_score(gt, pred, normalize=False)
    f1 = f1_score(gt, pred)*100.
    roc = roc_auc_score(gt, prob, average='macro')*100.
    p, r, _, _ = precision_recall_fscore_support(gt, pred, average='binary')
    # print p, r
    return acc, acc_not_normed, f1, roc, p, r


def merge_sliced_predictions(pred_csv_file):

    # 146745 [0.167403, 0.56373899999999999, 0.075844400000000006] 0.563739

    y_test, prob_pos, y_pred, noms, gt_dico_one_per_file = read_pred_csv_file_to_arrays(pred_csv_file)
    print len(noms)
    nb_predictions = len(noms)

    liste_noms = []
    concat_probs = {}

    ind=0
    while ind < nb_predictions:
        nom = noms[ind]
        if nom not in concat_probs.keys():
            concat_probs[nom] = []
            concat_probs[nom].append(prob_pos[ind])
        else:
            concat_probs[nom].append(prob_pos[ind])
        ind+=1

    print 'probs:', concat_probs['126544']
    assert len(concat_probs.keys()) == len(gt_dico_one_per_file.keys()), 'ERROR: not same number of keys in GT than in HYP'

    probs_one_per_file = []
    gt_one_per_file = []
    for nom, v in gt_dico_one_per_file.iteritems():
        gt_one_per_file.append(v)
        probs_one_per_file.append(max(concat_probs[nom]))
        # probs_one_per_file.append(sum(concat_probs[nom])/len(concat_probs[nom]))
        # if nom == '59234':
        #     print nom, v, concat_probs[nom], probs_one_per_file[-1]
        if nom == '126544' and probs_one_per_file[-1]>= 0.5:
            print nom, v, probs_one_per_file[-1], '\n'


    filelevel_probs = np.asarray(probs_one_per_file, dtype=np.float32)
    filelevel_preds = binary_predict(filelevel_probs, threshold = 0.5)
    print len(gt_one_per_file), filelevel_probs.shape, filelevel_preds.shape

    acc, acc_not_normed, f1, roc, p, r = acc_f1_roc(gt_one_per_file, filelevel_probs, filelevel_preds)
    print '#correct: %d acc: %.2f -- roc: %.2f -- p: %.2f r: %.2f f1: %.2f'%(acc_not_normed, acc, roc, p, r, f1)
