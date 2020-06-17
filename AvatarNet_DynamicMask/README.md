Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration
---

Usage
--

### Install requirement
```bash
pip install -e models/pytorch-image-models
```

and download pretrained weights at https://drive.google.com/file/d/14QxasSCcL_ij7NHR7Fshx5fi5Sc9MleD/view and place it in trained_models

### Webcam
```bash
python webcam.py --person_style path --background_style path --ratio num
```

### Arguments
* `--gpu-no`: GPU device number (-1: cpu, 0~N: GPU)
* `--train`: Flag for the network training (default: False)
* `--content-dir`: Path of the Content image dataset for training
* `--imsize`: Size for resizing input images (resize shorter side of the image)
* `--cropsize`: Size for crop input images (crop the image into squares)
* `--cencrop`: Flag for crop the center reigion of the image (default: randomly crop)
* `--check-point`: Check point path for loading trained network
* `--content`: Content image path to evalute the network
* `--style`: Style image path to evalute the network
* `--mask`: Mask image path for masked stylization
* `--style-strength`: Content vs Style interpolation weight (1.0: style, 0.0: content, default: 1.0)
* `--interpolatoin-weights`: Weights for multiple style interpolation
* `--patch-size`: Patch size of style decorator (default: 3)
* `--patch-stride`: Patch stride of style decorator (default: 1)

