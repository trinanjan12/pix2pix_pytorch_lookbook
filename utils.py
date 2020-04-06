import  os
from PIL import Image
import imageio

import torch
import torch.nn as nn
from  torchvision . utils  import  save_image
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np

#====================================================
# Save and load model
#====================================================
def save_checkpoint(model, device, save_path, step):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(
        {
            'step': step,
            'model_state_dict': model.cpu().state_dict(),
        }, save_path
    )
    model.to(device)
    return

def load_checkpoint(model, device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    step = checkpoint['step']
    model.to(device)
    return step

def load_checkpoint_wo_step(model, device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
        
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    return



def test_on_images(model, device, img_indexes,results_path,dloader,epoch_num):
    if not os.path.exists(os.path.join(results_path,'epoch_' + str(epoch_num))):
        os.mkdir(os.path.join(results_path,'epoch_' + str(epoch_num)))
    results_path = os.path.join(results_path,'epoch_' + str(epoch_num))
    for i in img_indexes:
        inputs_test = dloader[i]
        pre_image = inputs_test['source_image'].unsqueeze_(0).to(device)
        after_image = inputs_test['target_image'].unsqueeze_(0).to(device)
        out_pred = model(pre_image).squeeze(0).cpu()

        # scale all pixels from [-1,1] to [0,1]
        pre_image = pre_image[0].cpu().numpy()
        pre_image = (pre_image + 1) / 2
        pre_image = np.transpose(pre_image,(1,2,0))

        after_image = np.transpose(after_image[0].cpu().numpy(),(1,2,0))

        out_pred = out_pred.detach().cpu().numpy()
        out_pred = (out_pred + 1) / 2
        out_pred = np.transpose(out_pred,(1,2,0))

        # scale all pixels from [-1,1] to [0,1]
        vis = np.concatenate((pre_image,after_image,out_pred), axis=1)
        # plt.imshow(vis)
        Image.fromarray((vis * 255).astype('uint8')).save(os.path.join(results_path, 'epoch_' + str(epoch_num) + '_'  +str(i)+'.jpg'))
