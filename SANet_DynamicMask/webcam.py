import cv2
import time
import torch
import argparse
import numpy as np
import style_model
import torch.nn as nn
from models import UNet
from base import VideoInference
from torchvision import transforms

#---------------------------------Input arguments and some global settings----------------------------------#
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--human_style', type=str,
                    help='File path to the human style image')
parser.add_argument('--background_style', type=str,
                    help='File path to the background style image')
parser.add_argument('--ratio', type=float, default=0.5,
                    help='Style strength ratio')
args= parser.parse_args()

# Device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Path of pretrained weights of vgg encoder for style transfer
VGG_CHECKPOINT = "model_checkpoints/vgg_normalized.pth"
# Path of pretrained weights of decoder for style transfer
DECODER_CHECKPOINT = "model_checkpoints/decoder.pth"
# Path of pretrained weights of transformer for style transfer
TRANSFORMER_CHECKPOINT = "model_checkpoints/transformer.pth"
# Path of pretrained weights of UNet for segmentation
SEGMENTATION_NET_CHECKPOINT = "model_checkpoints/UNet_ResNet18.pth"

# Expected resolution of output video
OUT_WIDTH = int(16*35)      # The architecture of SANet requires that width and height of output video must be a multiple of 16
OUT_HEIGHT = int(16*20)

# Camera resolution
CAM_WIDTH = 1280
CAM_HEIGHT = 720
#-----------------------------------------------------------------------------------------------------------#


#----------------------------------------Util funcions------------------------------------------------------#
def central_square_crop(img):
    center = (img.shape[0] / 2, img.shape[1] / 2)
    h=w=min(img.shape[0],img.shape[1])
    x = center[1] - w/2
    y = center[0] - h/2
    crop_img = img[int(y):int(y+h), int(x):int(x+w),:]
    return crop_img

# Convert OpenCV imread image to RGB style and 0 ~ 1 range. The further convert it to normalized torch tensor with batch channel.
def toTensor(img):
    img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # Transpose from [H, W, C] -> [C, H, W]
    normalizer=transforms.Compose([transforms.ToTensor()])
    tensor = normalizer(img)
    # Add the batch_size dimension
    tensor = tensor.unsqueeze(0).to(device)
    return tensor

# Reverse the normalized image to normal distribution for displaying
def denormalize(img):
    MEAN = [0,0,0]
    STD = [1,1,1]
    MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
    STD = [1/std for std in STD]    
    denormalizer=transforms.Compose([transforms.Normalize(mean=MEAN, std=STD)])
    return denormalizer(img)

# Convert torch tensor back to OpenCV image for displaying
def toImage(tensor):
    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    tensor=denormalize(tensor)
    # Transpose from [C, H, W] -> [H, W, C]
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return img
#-----------------------------------------------------------------------------------------------------------#


#-----------------------------------------Model loading-----------------------------------------------------#

# Load segmentation net model
print("\nLoading Segmentation Network")
segmentation_model = UNet(backbone="resnet18", num_classes=2)
if torch.cuda.is_available():
    trained_dict = torch.load(SEGMENTATION_NET_CHECKPOINT)['state_dict']
else:
    trained_dict = torch.load(SEGMENTATION_NET_CHECKPOINT, map_location="cpu")['state_dict']
segmentation_model.load_state_dict(trained_dict, strict=False)
segmentation_model.to(device)
segmentation_model.eval()

# Create segmentation object
segmentObj = VideoInference(
    model=segmentation_model,
    video_path=0,
    input_size=320,
    height=OUT_HEIGHT,
    width=OUT_WIDTH,
    use_cuda=torch.cuda.is_available(),
    draw_mode='matting',
)
print("Done Loading Segmentation Network\n")

# Load style transfer net model
print("Loading Style Transfer Network")
decoder = style_model.decoder
transform = style_model.Transform(in_planes = 512)
vgg = style_model.vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(DECODER_CHECKPOINT))
transform.load_state_dict(torch.load(TRANSFORMER_CHECKPOINT))
vgg.load_state_dict(torch.load(VGG_CHECKPOINT))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)
print("Done Loading Style Transfer Network\n")

# Define style transfer function
def style_transfer(content_tensor,style_tensor):
    Content4_1 = enc_4(enc_3(enc_2(enc_1(content_tensor))))
    print(Content4_1.shape)
    Content5_1 = enc_5(Content4_1)
    print(Content5_1.shape)
    Style4_1 = enc_4(enc_3(enc_2(enc_1(style_tensor))))
    print(Style4_1.shape)
    Style5_1 = enc_5(Style4_1)
    print(Style5_1.shape)
    stylized_tensor = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))
    print(stylized_tensor.shape)
    return stylized_tensor

#-----------------------------------------------------------------------------------------------------------#


#------------------------------Human and background style image loading-------------------------------------#

# Load human style image and convert it to tensor
human_style_img=central_square_crop(cv2.imread(args.human_style))
human_style_img=cv2.resize(human_style_img,(512,512))
human_style_tensor=toTensor(human_style_img)

# Load background style image and convert it to tensor
background_style_img=central_square_crop(cv2.imread(args.background_style))
background_style_img=cv2.resize(background_style_img,(512,512))
background_style_tensor=toTensor(background_style_img)
#-----------------------------------------------------------------------------------------------------------#


#--------------------------------------Define web camera function-------------------------------------------#
def webcam():

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, CAM_WIDTH)
    cam.set(4, CAM_HEIGHT)

    # Main loop
    with torch.no_grad():
        while True:
            startTime=time.time()

            # Get webcam input
            ret_val, content_img = cam.read()

            # Mirror and resize
            content_img = cv2.flip(content_img, 1)
            content_img = cv2.resize(content_img,(OUT_WIDTH,OUT_HEIGHT),interpolation=cv2.INTER_CUBIC)
            content_tensor = toTensor(content_img)

            # Use segmentation to generate human_mask and background_mask
            # Shape of mask image is (H,W) and range is 0 ~ 255
            human_mask=np.round(segmentObj.run(content_img))
            background_mask=1-human_mask
            human_mask=np.array(255*human_mask,dtype=np.uint8)
            background_mask=np.array(255*background_mask,dtype=np.uint8)

            # Free-up unneeded cuda memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # stylize image
            with torch.no_grad():
                human_tensor = style_transfer(content_tensor, human_style_tensor)
                background_tensor = style_transfer(content_tensor, background_style_tensor)
            

            # Convert stylized tensors to images and resize    
            human_img = toImage(human_tensor)
            background_img = toImage(background_tensor)
            human_img = cv2.resize(human_img,(OUT_WIDTH,OUT_HEIGHT),interpolation=cv2.INTER_CUBIC)
            background_img = cv2.resize(background_img,(OUT_WIDTH,OUT_HEIGHT),interpolation=cv2.INTER_CUBIC)

            # Obtain regions of interest with mask and combine them to obtain final stylized image
            human_img = cv2.bitwise_and(human_img,human_img,mask=human_mask)
            background_img = cv2.bitwise_and(background_img,background_img,mask=background_mask) 
            stylized_img = human_img + background_img

            # Combine target style images 
            factor=0.5*content_img.shape[0]/human_style_img.shape[0]
            resized_human_style_img = cv2.resize(human_style_img,(0,0),fx=factor,fy=factor, interpolation = cv2.INTER_CUBIC)
            resized_background_style_img = cv2.resize(background_style_img,(0,0),fx=factor,fy=factor, interpolation = cv2.INTER_CUBIC)
            resized_style_img=np.vstack([resized_human_style_img,resized_background_style_img])

            # Concatenate targe style images, original image and stylized image
            # Don't convert to uint8 or there may be some display problem due to precision loss
            output=np.array(255*np.concatenate((resized_style_img/255,content_img/255,stylized_img),axis=1),dtype=np.uint8)
            output=np.concatenate((resized_style_img/255,content_img/255,stylized_img),axis=1)

            # Show webcam
            cv2.namedWindow('Semantic Recognized Real-time Camera Webcam Demo',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Semantic Recognized Real-time Camera Webcam Demo',int(0.5*OUT_HEIGHT+OUT_WIDTH),int(0.5*OUT_HEIGHT))
            cv2.imshow('Semantic Recognized Real-time Camera Webcam Demo', output)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit

            print("Generation time of last frame: {}".format(time.time()-startTime))
            
    # Free-up memories
    cam.release()
    cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------#

# Entrance
webcam()

