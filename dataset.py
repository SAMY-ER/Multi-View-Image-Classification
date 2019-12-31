import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CarPlugDataset(Dataset):
    """
    Custom Car Plug Dataset
    Returns tensor data items of shape (Views, Channels, Height, Width) with the corresponding 
    encoded label.
    """

    def __init__(self, root, plug_label_filename):
        self.root = root
        self.plug_label_map = self._get_plug_label_map(plug_label_filename)
        self.plug_names = self._get_plug_names(root)
        self.label_encoder = {'CPA Stecker':0, 
                              'Schraubstecker':1, 
                              'Bügelstecker':2, 
                              'Schieberstecker':3, 
                              'Kraftschlüssig mit Schraubsicherung':4, 
                              'Bajonette':5} 
        
    def _get_plug_names(self, root):
        plug_names = [fname.split('_')[0] for fname in os.listdir(root) if fname.endswith('.png')]
        plug_names = list(set(plug_names))
        return plug_names
    
    def _get_plug_label_map(self, filename):
        reader = csv.DictReader(open(filename))
        plug_label_map = {}
        for row in reader:
            plug_label_map[row['part_no']] = row['label']
        return plug_label_map
        
    def __len__(self):
        return len(self.plug_names)
    
    def _transform(self, image):
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        return transform(image)
    
    def __getitem__(self, index):
        plug_name = self.plug_names[index]
        # Get Images of the Plug
        plug_fnames = glob.glob(self.root + f'/{plug_name}_*.png')
        plug = torch.stack([self._transform(Image.open(fname).convert('RGB')) for fname in plug_fnames])
        label = self.label_encoder[self.plug_label_map[plug_name]]
        return plug, label