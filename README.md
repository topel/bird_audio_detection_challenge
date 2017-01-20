# Entry to the Bird Audio Detection challenge
based on Densely Connected Convolutional Networks (DenseNets) Theano/Lasagne

# About the challenge
To get information about the BAD challenge, please visit their [Website](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/)

# Usage
0. Feature extraction

Features: 56 log-Mel F-BANK coefficients, 58 bands, hop size: 50 ms, frame size: 100 ms, fmin: 50 Hz, fmax: 22050 Hz

   * MIR_extract_logSpectrumBands.m: extracts F-BANK coefficients from WAV files
   * create_hdf5_ff1010bird_public.py: creates an HDF5 file with Train, Valid and Test subsets
   
1. To train a densenet for 30 epochs:

```python
python train.py densenet 30
```

2. To test a model:

```python
python test.py densenet <modelpath> fbank
```

# Model Architecture
This code builds the following model. It is based on this [recipe](https://github.com/Lasagne/Recipes/tree/de347e97032569be017cc24319c471de92ac8b40/papers/densenet)

INFO: input layer: (None, 1, 200, 56)
INFO: first conv layer: (None, 32, 200, 56)
INFO: dense block 0: (None, 107, 200, 56)
INFO: transition 0: (None, 107, 100, 28)
INFO: dense block 1: (None, 182, 100, 28)
INFO: transition 1: (None, 182, 50, 14)
INFO: dense block 2: (None, 257, 50, 14)
INFO: post Global pool layer: (None, 257)
INFO: output layer: (None, 2)
INFO: total number of layers: 74
INFO: number of parameters in model: 328004

Each dense block corresponds to 5x[BatchNorm - ReLu - Conv3x3]
Each transition block corresponds to 1x[Conv1x1 - Max-Pool2x2]

