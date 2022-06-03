import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os


def make_dataset(image_list, label_list, au_relation=None):
    len_ = len(image_list)
    if au_relation is not None:
        images = [(image_list[i].strip(),  label_list[i, :],au_relation[i,:]) for i in range(len_)]
    else:
        images = [(image_list[i].strip(),  label_list[i, :]) for i in range(len_)]
    return images

def make_mtl_dataset(txt_file, au_relation=None):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        lines = [l.strip() for l in lines]
        lines = [l.split(',') for l in lines]

    path = [l[0] for l in lines]
    valence = [float(l[1]) for l in lines]
    arousal = [float(l[2]) for l in lines]
    expr = [int(l[3]) for l in lines]
    AUs = [np.array([float(x) for x in l[4:]]) for l in lines]

    ids_list = [i for i, x in enumerate(AUs)]
    AUs_new = [AUs[i] for i in ids_list]
    Val_new = [valence[i] for i in ids_list]
    Ars_new = [arousal[i] for i in ids_list]
    Exp_new = [expr[i] for i in ids_list]
    Pth_new = [path[i] for i in ids_list]

    if au_relation is not None:
        Rel_new = [au_relation[i, :] for i in ids_list]
        data_list = [(Pth_new[i], [Val_new[i], Ars_new[i]], Exp_new[i], AUs_new[i], Rel_new[i]) for i in range(len(AUs_new))]
    else:
        data_list = [(Pth_new[i], [Val_new[i], Ars_new[i]], Exp_new[i], AUs_new[i]) for i in range(len(AUs_new))]

    return data_list

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class BP4D(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'BP4D_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)


class DISFA(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'DISFA_train_img_path_fold' + str(fold) + '.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'DISFA_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path,img))

            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)

# s-AffWild2 Dataset
class SAW2(Dataset):
    def __init__(self, root_path, train=True, transform=None, stage=1, loader=default_loader):
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.loader = loader
        self.img_folder_path = os.path.join(root_path, 'cropped_aligned')
        if self._train:
            annotations_file = os.path.join(root_path, 'training_set_annotations.txt')

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'training_au_relation_annotations.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_mtl_dataset(annotations_file, au_relation_list)
            else:
                self.data_list = make_mtl_dataset(annotations_file)

        else:
            annotations_file = os.path.join(root_path, 'validation_set_annotations.txt')
            self.data_list = make_mtl_dataset(annotations_file)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, va, expr, au, rel = self.data_list[index]
        else:
            img, va, expr, au = self.data_list[index]
        img = self.loader(os.path.join(self.img_folder_path, img))

        if self._train:
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip)
        else:
            if self._transform is not None:
                img = self._transform(img)

        mask_va = int(-5 not in va)
        mask_expr = int(expr != -1)
        mask_au = int(-1 not in au)

        e = np.zeros(8)
        if expr != -1:
            e[expr] = 1

        if self._stage == 2 and self._train:
            return img, np.array(va), e, au, mask_va, mask_expr, mask_au, rel
        else:
            return img, np.array(va), e, au, mask_va, mask_expr, mask_au

    def __len__(self):
        return len(self.data_list)
