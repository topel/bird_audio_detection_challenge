
def set_options(feature_type):

    loss_type = 'categorical_crossentropy'
    # loss_type = 'binary_crossentropy'
    # loss_type='binary_hinge'
    # loss_type = 'weighted_binary_crossentropy'

    if loss_type == 'categorical_crossentropy':
        NB_CLASSES = 2
    else:
        NB_CLASSES = 1

    if feature_type=='fbank':
        options = {
            "NB_CHANNELS" : 1,
            "NB_FRAMES" : 200,
            "NB_FEATURES" : 56,
            "LOSS" : loss_type,
            "NB_CLASSES" : NB_CLASSES,
            "BATCH_SIZE" : 10,
            "CENTER_DATA" : False,
            "REDUCE_DATA" : False,
            "TEST_LABELS" : True, # are gt labels available?
            "AUGMENT" : True,
            "FEATURE_TYPE": feature_type
        }
    elif feature_type=='fbank_d_dd':
        options = {
            "NB_CHANNELS" : 3,
            "NB_FRAMES" : 200,
            "NB_FEATURES" : 56,
            "LOSS" : loss_type,
            "NB_CLASSES" : NB_CLASSES,
            "BATCH_SIZE" : 10,
            "CENTER_DATA" : False,
            "REDUCE_DATA" : False,
            "TEST_LABELS" : True, # are gt labels available?
            "AUGMENT" : True,
            "FEATURE_TYPE": feature_type
        }
    elif feature_type=='fp':
        options = {
            "NB_CHANNELS" : 1,
            "NB_FRAMES" : 192,
            "NB_FEATURES" : 200,
            "LOSS" : loss_type,
            "NB_CLASSES" : NB_CLASSES,
            "BATCH_SIZE" : 5,
            "CENTER_DATA" : False,
            "REDUCE_DATA" : False,
            "TEST_LABELS" : True, # are gt labels available?
            "AUGMENT" : True,
            "FEATURE_TYPE": feature_type
        }
    elif feature_type=='fp3':
        options = {
            "NB_CHANNELS" : 3,
            "NB_FRAMES" : 132,
            "NB_FEATURES" : 132,
            "LOSS" : loss_type,
            "NB_CLASSES" : NB_CLASSES,
            "BATCH_SIZE" : 10,
            "CENTER_DATA" : False,
            "REDUCE_DATA" : False,
            "TEST_LABELS" : True, # are gt labels available?
            "AUGMENT" : True,
            "FEATURE_TYPE": feature_type
        }
    elif feature_type=='ivec':
        options = {
            "NB_CHANNELS" : 1,
            "NB_FRAMES" : 1,
            "NB_FEATURES" : 600,
            "LOSS" : loss_type,
            "NB_CLASSES" : NB_CLASSES,
            "BATCH_SIZE" : 10,
            "CENTER_DATA" : False,
            "REDUCE_DATA" : False,
            "TEST_LABELS" : True, # are gt labels available?
            "AUGMENT" : False,
            "FEATURE_TYPE": feature_type
        }
    elif feature_type == 'fft':
        options = {
            "NB_CHANNELS" : 1,
            # "NB_FRAMES" : 10,
            "NB_FRAMES" : 430,
            "NB_FEATURES" : 512,
            "LOSS" : loss_type,
            "NB_CLASSES" : NB_CLASSES,
            "BATCH_SIZE" : 10,
            "CENTER_DATA" : False,
            "REDUCE_DATA" : False,
            "TEST_LABELS" : True, # are gt labels available?
            "AUGMENT" : False,
            "FEATURE_TYPE": feature_type
        }
    elif feature_type == 'slicedfft':
        options = {
            "NB_CHANNELS" : 1,
            # "NB_FRAMES" : 10,
            "NB_FRAMES" : 21,
            "NB_FEATURES" : 512,
            "LOSS" : loss_type,
            "NB_CLASSES" : NB_CLASSES,
            "BATCH_SIZE" : 100,
            "CENTER_DATA" : True,
            "REDUCE_DATA" : False,
            "TEST_LABELS" : True, # are gt labels available?
            "AUGMENT" : False,
            "FEATURE_TYPE": feature_type
        }
    elif feature_type == 'mfcc':
        options = {
            "NB_CHANNELS" : 1,
            # "NB_FRAMES" : 10,
            "NB_FRAMES" : 200,
            # "NB_FEATURES" : 13,
            "NB_FEATURES" : 56,
            "LOSS" : loss_type,
            "NB_CLASSES" : NB_CLASSES,
            "BATCH_SIZE" : 10,
            "CENTER_DATA" : True,
            "REDUCE_DATA" : False,
            "TEST_LABELS" : True, # are gt labels available?
            "AUGMENT" : False,
            "FEATURE_TYPE": feature_type
        }

    return options
