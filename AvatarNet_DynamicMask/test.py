import torch

from network import AvatarNet
from util import imload, imsave, maskload
import time
def network_test(args):
    startTime=time.time()

    # set device
    device = torch.device('cuda' if args.gpu_no >= 0 else 'cpu')
    
    # load check point
    if args.gpu_no >= 0:
        check_point = torch.load(args.check_point)
    else:
        check_point = torch.load(args.check_point,map_location=torch.device('cpu'))

    # load network
    network = AvatarNet(args.layers)
    network.load_state_dict(check_point['state_dict'])
    network = network.to(device)

    # load target images
    content_img = imload(args.content, args.imsize, args.cropsize).to(device)
    style_imgs = [imload(style, args.imsize, args.cropsize, args.cencrop).to(device) for style in args.style]
    masks = None
    if args.mask:
        masks = [maskload(mask).to(device) for mask in args.mask]

    # stylize image
    with torch.no_grad():
        stylized_img =  network(content_img, style_imgs, args.style_strength, args.patch_size, args.patch_stride,
                masks, args.interpolation_weights, False)
    

    imsave(stylized_img, 'stylized_image.jpg')

    print("Time elapsed: {}".format(time.time()-startTime))
    return None
