# coding=utf-8
import  os
import  argparse
import  numpy  as  np
from PIL import Image

import torch
import  torch . utils . data  as  data
import  torchvision . transforms  as  transforms
from  torchvision . utils  import  save_image

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class Map2AerialDataset(data.Dataset):
    """
    Aerial and map dataset classes
    """
    def __init__(self, root_dir, datamode = "train", image_height = 256, image_width = 256, debug = False ):
        super(Map2AerialDataset, self).__init__()

        # Specify the configuration of various pre-processing functions to be performed after loading data.
        self.transform = transforms.Compose(
            [
                transforms.Resize( (image_height, 2*image_width), interpolation=Image.LANCZOS ),
                transforms.CenterCrop( size = (image_height, 2 * image_width) ),
                transforms.ToTensor (),
                transforms.Normalize((0.5, ), (0.5, ))   # convert to Tensor
            ]
        )

        #
        self.image_height = image_height
        self.image_width = image_width
        self.dataset_dir = os.path.join( root_dir, datamode )
        self.image_names = sorted( [f for f in os.listdir(self.dataset_dir) if f.endswith(IMG_EXTENSIONS)] )
        self.debug = debug
        if( self.debug ):
            print( "self.dataset_dir :", self.dataset_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.image_names[0:5] :", self.image_names[0:5])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        raw_image = Image.open(os.path.join(self.dataset_dir, image_name)).convert('RGB')
        #print( "raw_image.size",  raw_image.size )
        raw_image_tsr = self.transform(raw_image)
        #print( "raw_image_tsr.shape",  raw_image_tsr.shape )

        # In the training data, the satellite image is on the left and the map image is on the right.
        # torch.chunk (): Divide the passed Tensor into the specified number.
        aerial_image_tsr, map_image_tsr = torch.chunk( raw_image_tsr, chunks=2, dim=2 )
        
        results_dict = {
            "image_name" : image_name,
            "raw_image_tsr" : raw_image_tsr,
            "aerial_image_tsr" : aerial_image_tsr,
            "map_image_tsr" : map_image_tsr,
        }
        return results_dict


class Map2AerialDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True):
        super(Map2AerialDataLoader, self).__init__()
        self.data_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle
        )

        self.dataset = dataset
        self.batch_size = batch_size
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

class lookbookdataset(data.Dataset):
    """
    Dataset class
    """
    def __init__(self, file_name, root_dir, datamode = "train", image_height = 256, image_width = 256, debug = False ):
        super(lookbookdataset, self).__init__()

        self.transform = transforms.Compose(
            [
                transforms.Resize( (image_height, image_width), interpolation=Image.LANCZOS ),
                transforms.CenterCrop( size = (image_height, image_width) ),
                transforms.ToTensor (),
                transforms.Normalize((0.5, ), (0.5, ))   # convert to Tensor
            ]
        )

        #
        self.image_height = image_height
        self.image_width = image_width
        self.dataset_dir = root_dir
        
        with open(file_name, 'r') as f:
            self.image_list = f.readlines()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        source_path = os.path.join(self.dataset_dir,self.image_list[index].split('\t')[0])
        target_path = os.path.join(self.dataset_dir + self.image_list[index].split('\t')[1].split('\n')[0])
        
        source_image = Image.open(source_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')
        
        source_image = self.transform(source_image)
        target_image = self.transform(target_image)

        results_dict = {
            "source_path" : source_path,
            "target_path" : target_path,
            "source_image" : source_image,
            "target_image" : target_image
        }
        
        return results_dict
