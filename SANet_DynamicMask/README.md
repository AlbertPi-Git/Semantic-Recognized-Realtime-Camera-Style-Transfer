# SANET

This is unofficial PyTorch implementation of "Arbitrary Style Transfer with Style-Attentional Networks".

Official paper: https://arxiv.org/abs/1812.02342v5

To run, download the weights and place them in the folder with Eval.py. Links to weights on Yandex.Disk:

* decoder: https://yadi.sk/d/xsZ7j6FhK1dmfQ

* transformer: https://yadi.sk/d/GhQe3g_iRzLKMQ

* vgg_normalised: https://yadi.sk/d/7IrysY8q8dtneQ

Or, you can download the latest release. It contains all weights, codes and examples.

# How to evaluate

## Single Image

```bash
python Eval.py --content path --style path
```

## Webcam

```bash
python webcam.py --style path
```

# How to train

You can train your own SANet using Train.ipynb
