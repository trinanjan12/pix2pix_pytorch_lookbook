import  argparse
import  os
import random
# from datetime import datetime
import  numpy  as  np
from tqdm import tqdm as tqdm
from PIL import Image
import time

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn

import  torchvision       # image processing related
import  torchvision.transforms  as  transforms
from  torchvision.utils  import  save_image
from tensorboardX import SummaryWriter

# Self-made class
from  make_dataset  import  lookbookdataset
from networks import Pix2PixUNetGenerator, Pix2PixDiscriminator, Pix2PixPatchGANDiscriminator
from  losses  import  LSGANLoss
from utils import save_checkpoint, load_checkpoint ,test_on_images
# from utils import board_add_image, board_add_images
# from utils import save_image_historys_gif



parser = argparse.ArgumentParser()
parser.add_argument("--exper_name", default="Pix2Pix_train_lookbook", help="exp name")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="available devices (CPU or GPU)")
#parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') 
parser.add_argument('--dataset_dir', type=str, default="./lookbook/data/", help="dataset_dir for dataloader")
parser.add_argument('--results_dir', type=str, default="results", help="output directory of the generated image")
parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="model storage directory")
parser.add_argument('--load_checkpoints_dir', type=str, default="", help="model load directory")
parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard visualization")
parser.add_argument('--n_test', type=int, default=10000, help="test dataset step")
parser.add_argument('--n_epoches', type=int, default=500, help="number of epoch")
parser.add_argument('--batch_size', type=int, default=16, help="number of batch size")
parser.add_argument('--batch_size_test', type=int, default=4, help="test batch_size")
parser.add_argument('--lr', type=float, default=0.0002, help="default lr")
parser.add_argument('--beta1', type=float, default=0.5, help="adam beta1")
parser.add_argument('--beta2', type=float, default=0.999, help="adam beta2")
parser.add_argument('--image_size', type=int, default=256, help="size of the input image (pixel units)")
parser.add_argument('--gan_type', choices=['vanilla', 'lsgan', 'hinge'], default="lsgan", help="GAN loss type")
parser.add_argument('--lambda_gan', type=float, default=1.0, help="Adv loss Coefficient")
parser.add_argument('--lambda_l1', type=float, default=100.0, help="L1 coefficient values of regularization term")
parser.add_argument('--unetG_dropout', type=float, default=0.5, help="dropout rate as input noise to generator")
parser.add_argument('--n_fmaps', type=int, default=64, help="feature maps the number of")
parser.add_argument('--networkD_type', choices=['vanilla','PatchGAN' ], default="PatchGAN", help="GAN Discriminator")
parser.add_argument('--n_display_step', type=int, default=50, help="display interval to Tensorboard")
parser.add_argument('--n_display_test_step', type=int, default=500, help="display interval to tensorboard of test data")
parser.add_argument("--n_save_step", type=int, default=5000, help="model checkpoint storage interval of")
parser.add_argument("--seed", type=int, default=8, help="seed for the network")
parser.add_argument('--debug', action='store_true', help="debug mode enabled")
args = parser.parse_args()


# Output execution condition
print( "----------------------------------------------" )
print ( "Conditions" )
print( "----------------------------------------------" )
print("Start time:" , time.time() )
print( "PyTorch version :", torch.__version__ )
for key, value in vars(args).items():
    print('%s: %s' % (str(key), str(value)))
# 実行 Device の設定
if( args.device == "gpu" ):
    use_cuda = torch.cuda.is_available()
    if( use_cuda == True ):
        device = torch.device( "cuda" )
        #torch.cuda.set_device(args.gpu_ids[0])
        print("Execution device:" , device )
        print( "GPU name :", torch.cuda.get_device_name(device))
        print("torch.cuda.current_device() =", torch.cuda.current_device())
    else:
        print( "can't using gpu." )
        device = torch.device( "cpu" )
        print( "Execution device:", device)
else:
    device = torch.device( "cpu" )
    print("Execution device:", device)
print('-------------- End ----------------------------')

device = "gpu"
if( device == "gpu" ):
    use_cuda = torch.cuda.is_available()
    if( use_cuda == True ):
        device = torch.device( "cuda" )
        #torch.cuda.set_device(args.gpu_ids[0])
        print("Execution device:" , device )
        print( "GPU name :", torch.cuda.get_device_name(device))
        print("torch.cuda.current_device() =", torch.cuda.current_device())
    else:
        print( "can't using gpu." )
        device = torch.device( "cpu" )
        print( "Execution device:", device)
else:
    device = torch.device( "cpu" )
    print("Execution device:", device)
print('-------------- End ----------------------------')


# create dirs
if not( os.path.exists(args.results_dir) ):
    os.mkdir(args.results_dir)
if not( os.path.exists(os.path.join(args.results_dir, args.exper_name)) ):
    os.mkdir( os.path.join(args.results_dir, args.exper_name) )
if not( os.path.exists(args.tensorboard_dir) ):
    os.mkdir(args.tensorboard_dir)
if not( os.path.exists(args.save_checkpoints_dir) ):
    os.mkdir(args.save_checkpoints_dir)
if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name)) ):
    os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name) )
if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name, "G")) ):
    os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name, "G") )
if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name, "D")) ):
    os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name, "D") )
    
# for visualation
board_train = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name) )
board_test = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_test") )

# fix seed value
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(a=args.seed)

#======================================================================
    # Read or generate dataset
    # Pre-processing data
#======================================================================
ds_train = lookbookdataset("lookbook_train_pair.txt",args.dataset_dir, "train", args.image_size, args.image_size)
ds_test = lookbookdataset("lookbook_test_pair.txt",args.dataset_dir, "test", args.image_size, args.image_size)

dloader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True )
dloader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size_test, shuffle=False )



model_G = Pix2PixUNetGenerator()
model_G.weight_init(mean=0.0, std=0.02)
model_G = model_G.to(device)

from torchsummary import summary
summary(model_G , input_size=(3, 256, 256))

model_D = Pix2PixPatchGANDiscriminator()
model_D.weight_init(mean=0.0, std=0.02)
model_D = model_D.to(device)

summary(model_D , input_size=[(3, 256, 256),(3, 256, 256)])


#======================================================================
# optimizer settings
#======================================================================
optimizer_G = optim.Adam(
    params = model_G.parameters(),
    lr = args.lr, betas = (args.beta1,args.beta2)
)

optimizer_D = optim.Adam(
    params = model_D.parameters(),
    lr = args.lr, betas = (args.beta1,args.beta2)
)

#======================================================================
# loss functions
#======================================================================
loss_gan_fn = LSGANLoss(device)
# L1 loss
loss_l1_fn = torch.nn.L1Loss()

for step, inputs in enumerate( tqdm( dloader_train, desc = "minbatch iters" ) ):
    print(inputs['source_image'].shape)
    print(inputs['target_image'].shape)
    print(inputs.keys())
    break


img_test_indexes = random.sample(range(0,len(dloader_test)), 20)
print(img_test_indexes)

#======================================================================
# Model learning process
#======================================================================
# History of generated images during learning
fake_image_historys = []

print("starting training loop...")
# iterations = 0
n_print = 1

train_hist = {}
train_hist['net_d_losses'] = []
train_hist['net_g_losses'] = []
train_hist['per_epoch_ptime'] = []
train_hist['train_ptime'] = []

#-----------------------------
# Training for a few epochs
#-----------------------------
for epoch in tqdm(range(args.n_epoches), desc="Epoches"):
    net_d_losses = []
    net_g_losses = []
    epoch_start_time = time.time()
    # Retrieve 1 minibatch from DataLoader, mini-batch processing
    for step, inputs in enumerate(tqdm(dloader_train, desc="minbatch iters")):
        model_G.train()
        model_D.train()

        # Ignore in the last mini-batch loop if less than batch size
        # (Because of later calculation, shape mismatch)
        if inputs["source_image"].shape[0] != args.batch_size:
            break
        # iterations += args.batch_size  ## to save the checkpoint ste

        # Transfer mini-batch data to GPU
        pre_image = inputs["source_image"].to(device)
        after_image = inputs["target_image"].to(device)
        #save_image( pre_image, "pre_image.png" )
        #save_image( after_image, "after_image.png" )

        #====================================================
        # Fitting process of classifier D
        #====================================================
        # Enable the gradient calculation of the network of the discriminator D that was disabled.
        for param in model_D.parameters():
            param.requires_grad = True

        #----------------------------------------------------
        # Inject training data into model
        #----------------------------------------------------
        # Classifier output when a real image is input

        d_real = model_D(pre_image, after_image)
        if (args.debug and n_print > 0):
            print("d_real.size() :", d_real.size())

        # Fake image output from generator
        with torch.no_grad():  # Prevents generator G from being updated.
            g_fake_img = model_G(pre_image)  ## ?? it should be pre_image
            if (args.debug and n_print > 0):
                print("g_fake_img.size() :", g_fake_img.size())

        # Classifier output when fake image is input
        # detach and prevent the gradient from propagating to the generator through g_fake_img
        d_fake = model_D(g_fake_img.detach(), after_image)
        if (args.debug and n_print > 0):
            print("d_fake.size() :", d_fake.size())
        #----------------------------------------------------
        # Calculate the loss function
        #----------------------------------------------------
        loss_D, loss_D_real, loss_D_fake = loss_gan_fn.forward_D(
            d_real, d_fake)

        #----------------------------------------------------
        # Update network
        #----------------------------------------------------
        # Initialize the gradient to 0 (this initialization process is necessary because the gradient is added for each iteration)
        optimizer_D.zero_grad()
        # Gradient calculation
        loss_D.backward()
        # update weight according to set optimizer based on gradient calculated by backward ()
        optimizer_D.step()

        train_hist['net_d_losses'].append(loss_D.item())
        net_d_losses.append(loss_D.item())
        #====================================================
        # Fitting process of generator G
        #====================================================
        # Do not calculate the gradient of the network of classifier D.
        for param in model_D.parameters():
            param.requires_grad = False
        #----------------------------------------------------
        # Inject training data into model
        #----------------------------------------------------
        # Fake image output from generator
        g_fake_img = model_G(pre_image)
        if (args.debug and n_print > 0):
            print("g_fake_img.size() :", g_fake_img.size())
        # Classifier output when fake image is input
        with torch.no_grad():
            d_fake = model_D(g_fake_img, after_image)
            if (args.debug and n_print > 0):
                print("d_fake.size() :", d_fake.size())
        #----------------------------------------------------
        # Calculate the loss function
        #----------------------------------------------------
        loss_gan = loss_gan_fn.forward_G(d_fake.detach())
        loss_l1 = loss_l1_fn(g_fake_img, after_image)

        # total
        loss_G = args.lambda_gan * loss_gan + args.lambda_l1 * loss_l1  ## ??

        #----------------------------------------------------
        # Update network
        #----------------------------------------------------
        # Initialize the gradient to 0 (this initialization process is necessary because the gradient is added for each iteration)
        optimizer_G.zero_grad()

        # Gradient calculation
        loss_G.backward()

        # update weight according to set optimizer based on gradient calculated by backward ()
        optimizer_G.step()

        train_hist['net_g_losses'].append(loss_G.item())
        net_g_losses.append(loss_G.item())
#         print('%d,%d,%d',(steps,loss_D, loss_G)
    if (epoch % 1 == 0):        
        test_on_images(model_G,device,img_test_indexes,os.path.join(args.results_dir, args.exper_name),ds_test,epoch)
        print("saving best model after 2 epoch")
        save_checkpoint( model_G, device,  os.path.join(args.save_checkpoints_dir, args.exper_name, "G", "epoch_" + str(epoch) + '_G_final.pth'), epoch )
        save_checkpoint( model_D, device,  os.path.join(args.save_checkpoints_dir, args.exper_name, "D", "epoch_" + str(epoch) + '_D_final.pth'), epoch )
        print( "saved checkpoints" )
    print('----------------------------------------------------------------')

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' %
          ((epoch + 1), args.n_epoches, per_epoch_ptime,torch.mean(torch.FloatTensor(net_d_losses)),
           torch.mean(torch.FloatTensor(net_g_losses))))
