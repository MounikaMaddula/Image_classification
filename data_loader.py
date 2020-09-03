
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os.path import basename
import glob 
import random

class RandomRotation():
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class Monkey_Dataset(Dataset):

    def __init__(self, root_folder, mode = 'training'):

        self.root_dir = root_folder + '/' + mode +'/'+ mode
        self.mode = mode

        self.images = glob.glob(self.root_dir+'/*/*.jpg')

        self.data_augment =[transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0),  \
                            transforms.CenterCrop(300),  \
                            transforms.Pad(20, fill=0),  \
                            RandomRotation(angles=[-30, -15, 15, 30]),  \
                            transforms.RandomHorizontalFlip()  \
                            ]

        if self.mode == 'train':
            prob = 0.5
        else :
            prob = 0

        self.transforms = transforms.Compose([  \
                            transforms.RandomApply([transforms.RandomChoice(self.data_augment)], p=prob),  \
                            transforms.Resize((224,224)),  \
                            transforms.ToTensor(),  \
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  \
                            ])

    def __getitem__(self, ix):

        img = Image.open(self.images[ix])
        img = self.transforms(img)
        label = int(basename(self.images[ix])[1:2])
        return img, label

    def __len__(self):
        return len(self.images)

def main():
    monkey_data = Monkey_Dataset(root_folder = '../monkey_challenge')
    
    for i in range(len(monkey_data)) :
        img, label = monkey_data[i]
        print (img.shape, label)

if __name__ == '__main__':
    main()