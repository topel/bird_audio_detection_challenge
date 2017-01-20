# Entry to the Bird Audio Detection challenge
Based on Densely Connected Convolutional Networks (DenseNets) Theano/Lasagne

## About the BAD challenge
To get information about the challenge, please visit its [Website](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/)

## Requirements
* Theano (0.9.0.dev3)
* Lasagne (0.2.dev1)
* h5py (2.6.0)
* MIR toolbox for feature extraction with Matlab


## Usage
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

## Model Architecture
The code builds the following model. It is based on this [recipe](https://github.com/Lasagne/Recipes/tree/de347e97032569be017cc24319c471de92ac8b40/papers/densenet)

![Model Image](bird_audio_detection_challenge/densenet.png)


input layer: (None, 1, 200, 56)<br/>
first conv layer: (None, 32, 200, 56)<br/>
dense block 0: (None, 107, 200, 56)<br/>
transition 0: (None, 107, 100, 28)<br/>
dense block 1: (None, 182, 100, 28)<br/>
transition 1: (None, 182, 50, 14)<br/>
dense block 2: (None, 257, 50, 14)<br/>
post Global pool layer: (None, 257)<br/>
output layer: (None, 2)

total number of layers: 74<br/>
number of parameters in model: 328004<br/>

Each dense block corresponds to 5x[BatchNorm - ReLu - Conv3x3]<br/>
Each transition block corresponds to 1x[Conv1x1 - Max-Pool2x2]<br/>

