import os
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader, listdir_nohidden
from collections import OrderedDict


class ImageNetSketch(DatasetBase):
    """ImageNet-Sketch.

    This dataset is used for testing only.
    """

    dataset_dir = 'imagenet-sketch'

    def __init__(self, root):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        text_file = os.path.join(self.dataset_dir, 'classnames.txt')
        classnames = self.read_classnames(text_file)
        
        data = self.read_data(classnames)

        super().__init__(train_x=data, test=data)
    
    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)
        
        return items
    
    def read_classnames(self, text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                folder = line[0]
                classname = ' '.join(line[1:])
                classnames[folder] = classname
        return classnames
