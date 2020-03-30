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
        aerial_image_tsr = inputs_test['source_image'].unsqueeze_(0).to(device)
        map_image_tsr = inputs_test['target_image'].unsqueeze_(0).to(device)
        out_test = model(aerial_image_tsr).squeeze(0).cpu()
        test_1 = np.transpose(aerial_image_tsr[0].cpu().numpy(),(1,2,0))
        test_label_1 = np.transpose(map_image_tsr[0].cpu().numpy(),(1,2,0))
        out_test = out_test.detach().cpu().numpy()
        out_test = np.transpose(out_test,(1,2,0))
        vis = np.concatenate((test_label_1,test_1, out_test), axis=1)
#         plt.imshow(vis)
        Image.fromarray((vis * 255).astype('uint8')).save(os.path.join(results_path, 'epoch_' + str(epoch_num) + '_'  +str(i)+'.jpg'))
