# PyTorch
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Pix2PixUNetGenerator(nn.Module):
    """
    UNet generator
    """
    def __init__(self,
                 n_in_channels=3,
                 n_out_channels=3,
                 n_fmaps=64,
                 dropout=0.5):
        # Input noise z input to generator G is realized as noise in the sense that dropout is applied directly.
        super(Pix2PixUNetGenerator, self).__init__()

        def conv_block(in_dim, out_dim):
            model = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim),nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout))
            return model
        
        
        
        def conv_block_first(in_dim, out_dim):
            model = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout))
            return model

        def dconv_block(in_dim, out_dim):
            model = nn.Sequential(
                nn.ConvTranspose2d(in_dim,
                                   out_dim,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1), nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout))
            return model
    
        # Encoder (down sampling)
        self.conv1 = conv_block_first(n_in_channels, n_fmaps) ## (bs,3,256,256) --> (bs,64,256,256)
        self.pool1 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0) ## (bs,64,256,256) --> (bs,64,128,128)
        self.conv2 = conv_block( n_fmaps*1, n_fmaps*2 ) ## (bs,64,128,128) --> (bs,128,128,128)
        self.pool2 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0) ## (bs,128,128,128) --> (bs,128,64,64)
        self.conv3 = conv_block( n_fmaps*2, n_fmaps*4 ) ## bs,128,64,64) --> (bs,256,64,64)
        self.pool3 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0) ## (bs,256,64,64) --> (bs,256,32,32)
        self.conv4 = conv_block( n_fmaps*4, n_fmaps*8 ) ## (bs,256,32,32) --> (bs,512,32,32)
        self.pool4 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0) ## (bs,512,32,32) --> (bs,512,16,16)

        self.bridge= conv_block(n_fmaps*8, n_fmaps*16 ) ## (bs,512,16,16) --> (bs,1024,16,16)
        
        self.dconv1 = dconv_block(n_fmaps*16 , n_fmaps*8) ## (bs, 1024, 16, 16) --> (bs,512,32,32)
        self.up1 = conv_block(n_fmaps*16 , n_fmaps*8) ## (bs,1024,32,32) (512 + 512) --> (bs,512,32,32) 
        self.dconv2 = dconv_block(n_fmaps*8 , n_fmaps*4) ## (bs,512,32,32)  --> (bs,256,64,64) 
        self.up2 = conv_block(n_fmaps*8 , n_fmaps*4) ## (bs,512,64,64) (256 + 256) --> (bs,256,64,64) 
        self.dconv3 = dconv_block(n_fmaps*4 , n_fmaps*2) ## (bs,256,64,64)  --> (bs,128,128,128) 
        self.up3 = conv_block(n_fmaps*4 , n_fmaps*2) ## (bs,256,128,128) (128 + 128) --> (bs,128,128,128) 
        self.dconv4 = dconv_block( n_fmaps*2, n_fmaps*1 ) ## (bs,128,128,128)  --> (bs,64,256,256) 
        self.up4 = conv_block( n_fmaps*2, n_fmaps*1 ) ## (bs,128,256,256) (64 + 64) --> (bs,64,256,256) 
        
        self.out_layer = nn.Sequential(
            nn.Conv2d( n_fmaps, n_out_channels, 3, 1, 1 ),
            nn.Tanh()
        )
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward( self, input ):
        # Encoder
        conv1 = self.conv1( input )
        pool1 = self.pool1( conv1 )
        conv2 = self.conv2( pool1 )
        pool2 = self.pool2( conv2 )
        conv3 = self.conv3( pool2 )
        pool3 = self.pool3( conv3 )
        conv4 = self.conv4( pool3 )
        pool4 = self.pool4( conv4 )
        #
        bridge = self.bridge( pool4 )

        # Decoder（アップサンプリング）& skip connection
        dconv1 = self.dconv1(bridge)

        concat1 = torch.cat( [dconv1,conv4], dim=1 )
        up1 = self.up1(concat1)

        dconv2 = self.dconv2(up1)
        concat2 = torch.cat( [dconv2,conv3], dim=1 )

        up2 = self.up2(concat2)
        dconv3 = self.dconv3(up2)
        concat3 = torch.cat( [dconv3,conv2], dim=1 )

        up3 = self.up3(concat3)
        dconv4 = self.dconv4(up3)
        concat4 = torch.cat( [dconv4,conv1], dim=1 )

        up4 = self.up4(concat4)
        output = self.out_layer( up4 )
        
        return output
    
#====================================
# Discriminators
#====================================
class Pix2PixDiscriminator( nn.Module ):
    """
    A model describing the network configuration on the classifier side.
    
    """
    def __init__(self, n_channels=3,n_fmaps=64):
        super(Pix2PixDiscriminator,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(n_channels*2, n_fmaps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(n_fmaps, n_fmaps*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_fmaps*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_fmaps*2, n_fmaps*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_fmaps*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_fmaps*4, n_fmaps*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_fmaps*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_fmaps*8, 1, kernel_size=4, stride=1, padding=0, bias=False),

        )
        #weights_init( self )
        
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, x,y):
        output = torch.cat([x,y],dim = 1)
        output = self.layer(output)
        return output.view(-1)
    
class Pix2PixPatchGANDiscriminator( nn.Module ):
    """
    PatchGAN classifier
    """
    def __init__(
        self,
        n_in_channels = 3,
        n_fmaps  =  32
    ):
        super( Pix2PixPatchGANDiscriminator, self ).__init__()

        # In the classifier network, Patch GAN is adopted,
        # Do not cut or stride patches directly
        # Instead, express this by convolution.
        # In other words, 1 pixel with a feature map obtained by convolving CNN is a value affected by a certain area (Receptive field) of the input image,
        # In other words, only one area of ​​the input image can be affected by one pixel.
        # Therefore, "the final output is a feature map with a certain size, and the authenticity is determined at each pixel" and "the input image is a patch and the authenticity is determined at the output of each patch" This is because
        def discriminator_block1( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride=2, padding=1 ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        def discriminator_block2( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride=2, padding=1 ),
                nn.BatchNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        self.layer1 = discriminator_block1( n_in_channels * 2, n_fmaps )
        self.layer2 = discriminator_block2( n_fmaps, n_fmaps*2 )
        self.layer3 = discriminator_block2( n_fmaps*2, n_fmaps*4 )
        self.layer4 = discriminator_block2( n_fmaps*4, n_fmaps*8 )

        self.output_layer = nn.Sequential(
            nn.ZeroPad2d( (1, 0, 1, 0) ),
            nn.Conv2d( n_fmaps*8, 1, 4, padding=1, bias=False )
        )
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x, y ):
        output = torch.cat( [x, y], dim=1 )
        output = self.layer1( output )
        output = self.layer2( output )
        output = self.layer3( output )
        output = self.layer4( output )
        output = self.output_layer( output )
        output = output.view(-1)
        return output
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()