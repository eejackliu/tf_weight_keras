# import tensorflow as tf
import keras
import os
import math
from PIL import Image as image
import numpy as np
voc_colormap = [[0, 0, 0], [245,222,179]]
class id_data(keras.utils.Sequence):
    def __init__(self,root='/media/llm/D08E2AF4DD65FFFC/data/',image_set='train',batch_size=2,temp=1):
        super(id_data,self).__init__()
        self.root=root
        self.batch_size=batch_size
        image_dir=os.path.join(self.root,'img')
        mask_dir=os.path.join(self.root,'mask')
        self.image_set='id_'+image_set
        splits_f=os.path.join('/home/llm/PycharmProjects/tf_weight_keras', self.image_set + '.txt')
        with open(os.path.join(splits_f),'r') as f:
            self.file_name=[x.strip() for x in f.readlines()]

        self.image=[os.path.join(image_dir,x+'.tif') for x in self.file_name]
        self.mask=[os.path.join(mask_dir,x+'.jpg') for x in self.file_name]
        # if self.image_set=='train':
        #     self.softmask=np.load('/home/llm/PycharmProjects/fulldeeplab/logits2.npy')/temp
        #     self.softmask=1/(1+np.exp(-self.softmask))
        assert (len(self.image)==len(self.mask))
    def __len__(self):
        return math.ceil(len(self.file_name)/self.batch_size)
    def __getitem__(self, item):
        mean,std=np.array([0.485, 0.456, 0.406]),np.array((0.229, 0.224, 0.225))
        # mean,std=np.array([123, 117, 104]),np.array()
        try:
           img=self.image[item*self.batch_size:(item+1)*self.batch_size]
           mask=self.mask[item*self.batch_size:(item+1)*self.batch_size]

           mask=np.array([(np.array(image.open(i).convert('L'))/255.0>0.3).astype(int) for i in mask])  # origin is rgba turn it to gray scale
           # little_mask=np.array([np.array(image.open(i).resize((30,40)))/255 for i in mask])
           mask_t=mask[:,:,:,None]
           img_t=np.array([np.array(image.open(i).convert('RGB')) for i in img]).astype(np.float32)/255.
           img_t=(img_t-mean)/std
           # img_t=np.array([np.array(image.open(i).convert('RGB')) for i in img]).astype(np.float32)
        except Exception as e:
            print(e)
        # if self.image_set=='test':
        #     return img_t,mask_t
        if hasattr(self,'softmask'):
            softmask=self.softmask[item*self.batch_size:(item+1)*self.batch_size]

            return img_t,[softmask,mask_t]
        else:
            return img_t,mask_t


def hist(label_true,label_pred,num_cls):
    # mask=(label_true>=0)&(label_true<num_cls)
    hist=np.bincount(label_pred.astype(int)*num_cls+label_true.astype(int),minlength=num_cls**2).reshape(num_cls,num_cls)
    return hist
def label_acc_score(label_true,label_pred,num_cls=2):
    hist_matrix=np.zeros((num_cls,num_cls))
    tmp=0
    for i,j in zip(label_true,label_pred):
        hist_matrix+=hist(i.flatten(),j.flatten(),num_cls)
        tmp+=1
    diag=np.diag(hist_matrix)
    # acc=diag/hist_matrix.sum()
    acc_cls=diag/hist_matrix.sum(axis=0)
    m_iou=diag/(hist_matrix.sum(axis=1)+hist_matrix.sum(axis=0)-diag)
    return acc_cls,m_iou,hist_matrix,tmp