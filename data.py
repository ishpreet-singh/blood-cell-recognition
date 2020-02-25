import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split


this_path = os.path.dirname(os.path.realpath(__file__))


class Data:
    def __init__(self, batch_size):
        self.in_ht, self.in_wd = 240, 320
        self.create_stat_image = False
        # self.create_stat_image = True
        self.out_ht, self.out_wd = int(self.in_ht / 2), int(self.in_wd / 2)
        self.vld_portion = 0.1
        self.batch_size = {'TRAIN': batch_size, 'VALIDATION': batch_size, 'TEST': 1}
        self.in_dir = os.path.join(this_path, 'dataset', 'dataset2-master', 'images')

        # Get all cell names directories, ['NEUTROPHIL', 'MONOCYTE', 'EOSINOPHIL', 'LYMPHOCYTE']
        self.id2cell = pd.Series(os.listdir(os.path.join(self.in_dir, 'TRAIN')))
        self.cell2id = pd.Series(range(len(self.id2cell)), index=self.id2cell)

        # x_trn_list -> all file names in training, 8961
        # x_vld_list -> all file names in validation, 996
        # y_vld -> labelled output of size num of training data * num of classes, 8961 * 4
        # y_vld -> labelled output of size num of validation data * num of classes, 996 * 4
        self.x_trn_list, self.x_vld_list, self.y_trn, self.y_vld = self._get_names_labels(phase='TRAIN')
        
        # x_tst_list -> all file names in Testing set, 2487 
        # y_tst -> labelled output of size num of Testing data * num of classes, 2487 * 4
        self.x_tst_list, self.y_tst = self._get_names_labels(phase='TEST')
        
        # steps_per_epoch -> Steps per epoch in Training, 71
        self.steps_per_epoch = int(np.ceil(len(self.x_trn_list)/self.batch_size['TRAIN']))
        # validation_steps -> Steps per epoch in Training, 8
        self.validation_steps = int(np.ceil(len(self.x_vld_list)/self.batch_size['TRAIN']))
        # test_steps -> Steps per epoch in Training, 2487
        self.test_steps = int(np.ceil(len(self.x_tst_list)/self.batch_size['TEST']))

        # Creating Mean and Standard Image, Why they are created -_-?
        self.mean_img, self.std_img = self._get_stat_images()

    def _get_names_labels(self, phase):
        if phase not in ('TRAIN', 'TEST'):
            raise self.DataInitError("Error: 'phase' must be either of 'TRAIN' or 'TEST'")

        in_dir = os.path.join(self.in_dir, phase)
        if not os.path.exists(in_dir):
            raise self.DataInitError('Error: Directory {:s} does not exist.'.format(in_dir))

        x = list()
        labels = dict()
        for cell_id in self.id2cell.index:
            img_dir = os.path.join(in_dir, self.id2cell[cell_id])
            img_names = [a for a in os.listdir(img_dir) if a.endswith('.jpeg')]
            img_paths = [os.path.join(img_dir, img_name) for img_name in img_names]
            x += img_paths
            labels[cell_id] = np.zeros([len(img_paths), len(self.id2cell)], dtype=bool)  # One hot vector
            labels[cell_id][:, cell_id] = True

        y = np.concatenate([labels[a] for a in self.id2cell.index])

        if phase == 'TRAIN':
            trn_x_list, vld_x_list, y_trn, y_vld = \
                train_test_split(x, y, test_size=self.vld_portion, random_state=42, stratify=y, shuffle=True)

            return trn_x_list, vld_x_list, y_trn, y_vld
        else:
            return x, y

    def _normalize_image(self, img_in):
        if 0:  # 100_4_10
            # Unnormalized
            img = img_in
        elif 0:  # 100_4_11
            # Normalize each image independently
            img = np.empty(img_in.shape, dtype=img_in.dtype)
            assert(img_in.shape[2] == 3)
            for i in range(img_in.shape[2]):  # Loop over channels
                # Within an image, normalize each channel independently
                img[:, :, i] = (img_in[:, :, i] - img_in[:, :, i].mean()) / img_in[:, :, i].std()
        elif 0:  # 100_4_12
            # Just scale range (0, 255) to (-1, +1)
            img = img_in/255. - 0.5
        else:  # 100_4
            # Default: Normalize each dimension
            img = (img_in - self.mean_img) / self.std_img

        return img

    def get_batch(self, phase):
        if phase not in ('TRAIN', 'TEST', 'VALIDATION'):
            raise self.DataBatchError("Error: 'phase' must be either of 'TRAIN', 'TEST' or 'VALIDATION")

        if phase == 'TRAIN':
            x_list = self.x_trn_list
            y = self.y_trn
        elif phase == 'VALIDATION':
            x_list = self.x_vld_list
            y = self.y_vld
        else:
            x_list = self.x_tst_list
            y = self.y_tst

        # Allocated one-time memory for the batch
        x_batch = np.zeros((self.batch_size[phase], self.out_ht, self.out_wd, 3), dtype=float)
        y_batch = np.zeros((self.batch_size[phase], len(self.cell2id)), dtype=bool)

        src_idx = 0
        dst_idx = 0
        while True:
            img_path = x_list[src_idx]
            img = cv2.imread(img_path)
            if img is None:
                raise self.DataBatchError("Error: Can't open image: {:s}".format(img_path))

            img = cv2.resize(img, (self.out_wd, self.out_ht)).astype(float)

            # Normalize the image
            img = self._normalize_image(img)

            x_batch[dst_idx] = img
            y_batch[dst_idx] = y[src_idx]
            src_idx += 1
            dst_idx += 1

            if src_idx >= len(x_list):
                src_idx = 0

            if dst_idx >= self.batch_size[phase]:
                dst_idx = 0
                yield x_batch.copy(), y_batch.copy()

    def _get_stat_images(self):
        mean_img_path = os.path.join(this_path, 'resources', 'mean_image_{:d}x{:d}.npy'.format(self.out_wd, self.out_ht))
        std_img_path = os.path.join(this_path, 'resources', 'std_image_{:d}x{:d}.npy'.format(self.out_wd, self.out_ht))

        if self.create_stat_image:
            print("Creating Mean and Std images... ")
            x_train = np.empty((len(self.x_trn_list), self.out_ht, self.out_wd, 3), dtype=np.uint8)
            for idx, img_path in enumerate(self.x_trn_list):
                img = cv2.imread(img_path)
                if img is None:
                    raise self.DataInitError("Error: Can't open image: {:s}".format(img_path))

                img = cv2.resize(img, (self.out_wd, self.out_ht))
                x_train[idx] = img

            mean_img = x_train.mean(axis=0)
            std_img = x_train.std(axis=0)
            np.save(mean_img_path, mean_img)
            np.save(std_img_path, std_img)
            # Dump images as well for verification purposes
            cv2.imwrite(mean_img_path.replace('.npy', '.png'), mean_img.astype(np.uint8))
            cv2.imwrite(std_img_path.replace('.npy', '.png'), std_img.astype(np.uint8))

            print('Done writing mean and std images. ')

        mean_img = np.load(mean_img_path)
        std_img = np.load(std_img_path)
        return mean_img, std_img

    def read_dir(self, dir_path, id2cell, stat_imgs=None):
        if not os.path.exists(dir_path):
            raise self.DataInitError('Error: Directory {:s} does not exist.'.format(dir_path))

        imgs = dict()
        labels = dict()

        if stat_imgs is not None:
            mean_img = stat_imgs[0]
            std_img = stat_imgs[1]

        for id in id2cell.index:
            img_dir = os.path.join(dir_path, id2cell[id])
            img_names = [x for x in os.listdir(img_dir) if x.endswith('.jpeg')]
            imgs[id] = np.zeros([len(img_names), self.out_ht, self.out_wd, 3], dtype=float)

            for i, img_name in enumerate(img_names):
                img = cv2.imread(os.path.join(img_dir, img_name))
                assert(img is not None)
                img = cv2.resize(img, (self.out_wd, self.out_ht)).astype(float)
                if stat_imgs is not None:  # Normalizing images
                    img = (img - mean_img)/std_img
                imgs[id][i, :, :, :] = img

            labels[id] = np.zeros([len(imgs[id]), len(id2cell)], dtype=bool)  # One hot vector
            labels[id][:, id] = True

        x = np.concatenate([imgs[a] for a in id2cell.index])
        y = np.concatenate([labels[a] for a in id2cell.index])
        return x, y

    def load_data(self, create_stat_image=False):

        in_dir = os.path.join(this_path, 'dataset', 'dataset2-master', 'images')

        id2cell = pd.Series(os.listdir(os.path.join(in_dir, 'TRAIN')))
        cell2id = pd.Series(range(len(id2cell)), index=id2cell)

        mean_img_path = os.path.join(this_path, 'resources', 'mean_image_{:d}x{:d}.npy'.format(self.out_wd, self.out_ht))
        std_img_path = os.path.join(this_path, 'resources', 'std_image_{:d}x{:d}.npy'.format(self.out_wd, self.out_ht))

        if create_stat_image:
            x_train, y_train = self.read_dir(os.path.join(in_dir, 'TRAIN'), id2cell)
            x_test, y_test = self.read_dir(os.path.join(in_dir, 'TEST'), id2cell)

            x_train, x_validation, y_train, y_validation = \
                train_test_split(x_train, y_train, test_size=self.vld_portion, random_state=42, stratify=y_train)

            mean_img = x_train.mean(axis=0)
            std_img = x_train.std(axis=0)
            print('Writing mean and std images.. ')
            np.save(mean_img_path, mean_img)
            np.save(std_img_path, std_img)
            exit(0)
        else:
            mean_img = np.load(mean_img_path)
            std_img = np.load(std_img_path)
            assert((mean_img is not None) and (std_img is not None))
            x_train, y_train = self.read_dir(os.path.join(in_dir, 'TRAIN'), id2cell, (mean_img, std_img))
            x_test, y_test = self.read_dir(os.path.join(in_dir, 'TEST'), id2cell, (mean_img, std_img))

            x_train, x_validation, y_train, y_validation = \
                train_test_split(x_train, y_train, test_size=self.vld_portion, random_state=42, stratify=y_train)

        return x_train, x_validation, x_test, y_train, y_validation, y_test


    class DataError(Exception):
        """ Base class for all exceptions """
        pass

    class DataInitError(DataError):
        """ Initialization failed """
        pass

    class DataBatchError(DataError):
        """ Failed while getting next batch """
        pass












