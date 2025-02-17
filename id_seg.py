
import  tensorflow as tf

import numpy as np
import keras
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.engine import Layer
from keras.layers import ReLU
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from id_data import label_acc_score,id_data
# import keras
import glob
import math
import  matplotlib.pyplot as plt
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    # pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    pointwise_filters=pointwise_conv_filters
    x = inputs
    prefix = 'expanded_conv_{}/'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand/BatchNorm')(x)
        # x = Activation(relu6, name=prefix + 'expand_relu')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv/'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise/BatchNorm')(x)

    # x = Activation(relu6, name=prefix + 'depthwise_relu')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)
    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project/BatchNorm')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x

def mDeeplab(weights='pascal_voc', input_tensor=None, input_shape=(320, 184, 3), classes=1, backbone='mobilenetv2', OS=8, alpha=0.5):
    """ Instantiates the Deeplabv3+ architecture

    Optionally loads weights pre-trained
    on PASCAL VOC. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc)
            or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        classes: number of desired classes. If classes != 21,
            last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone

    # Returns
        A Keras model instance.

    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`

    """


    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor



    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv',)(img_input)

    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm')(x)
    # x = Activation(relu6, name='Conv_Relu6')(x)
    x = ReLU(6., name='Conv_Relu6')(x)
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling



    # Image Feature branch
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(40,23))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling/BatchNorm', epsilon=1e-5)(b4)
    # b4 = Activation('relu')(b4)
    b4 = ReLU(6.)(b4)
    # b4 = BilinearUpsampling((40,30))(b4)
    # b4=K.tf.image.resize_bilinear(b4, (40,30),align_corners=True)
    b4=keras.layers.UpSampling2D((40,23),interpolation='bilinear')(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0/BatchNorm', epsilon=1e-5)(b0)
    # b0 = Activation('relu', name='aspp0_activation')(b0)
    b0 = ReLU()(b0)

    x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection/BatchNorm', epsilon=1e-5)(x)
    # x = Activation('relu')(x)
    x= ReLU()(x)
    x = Dropout(0.1)(x)

    # DeepLab v.3+ decoder

    x = Conv2D(classes, (1, 1), padding='same', name='custom_layer')(x)
    # x = BilinearUpsampling(output_size=(320,240))(x)
    # x=K.tf.image.resize_bilinear(x, (320, 240), align_corners=True)
    x = keras.layers.UpSampling2D((8, 8), interpolation='bilinear',name='logit_label')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor,name='main_input')
    else:
        inputs = img_input
    x = keras.layers.Activation('sigmoid',name='stand')(x)
    # input=keras.layers.Input(shape=(320,240,3),name='main_input')
    model = Model(inputs=inputs, outputs=x)

    # load weight
    # model.load_weights('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5', by_name=True)



    return model
# model=mDeeplab()
def model_initialize():
    model=mDeeplab()
    tf_name=list(np.load('name_change.npy'))
    tf_value=np.load('value.npy')
    f=open('success','w')
    for k,i in enumerate(model.layers):
        tmp='MobilenetV2/'+i.name+'/weights'
        tbn='MobilenetV2/'+i.name+'/gamma'

        upconv=i.name+'/weights'
        upbn=i.name+'/gamma'
        if k<147:
            if tmp in tf_name:
                index=tf_name.index(tmp)
                i.set_weights([tf_value[index]])
                f.writelines(str(k)+' '+i.name+"  ok    "+str(len(i.get_weights()))+'\n')
            elif tbn in tf_name:
                index=tf_name.index(tbn)
                data_=tf_value[index:index+9]
                data=[ j  for j in data_ if j is not None]
                i.set_weights(data)
                f.writelines(str(k) + ' ' + i.name + "  ok    " + str(len(i.get_weights())) + '\n')
            else:
                f.writelines(str(k)+' '+i.name+"  not initialize    "+str(len(i.get_weights()))+'\n')
                if len(i.get_weights())>0:
                    print (i.name+"  not initialize")
        else:
            if upconv in tf_name:
                index=tf_name.index(upconv)
                i.set_weights([tf_value[index]])
                f.writelines(str(k)+' '+i.name+"  ok    "+str(len(i.get_weights()))+'\n')
            elif upbn in tf_name:
                index=tf_name.index(upbn)
                data_=tf_value[index:index+9]
                data=[ j  for j in data_ if j is not None]
                i.set_weights(data)
                f.writelines(str(k) + ' ' + i.name + "  ok    " + str(len(i.get_weights())) + '\n')

            else:
                f.writelines(str(k)+' '+i.name+"  not initialize    "+str(len(i.get_weights()))+'\n')
                if len(i.get_weights())>0:
                    print (i.name+"  not initialize")
    f.close()
    return model
def iou(y_true,y_pred):
    y_pred=(y_pred>0.5)
    y_pred=keras.backend.cast(y_pred, dtype='float32')
    return keras.backend.sum(y_true*y_pred)/(keras.backend.sum(y_pred),keras.backend.sum(y_true)-keras.backend.sum(y_true*y_pred)+0.0001)
def diceloss(y_true,y_pred):

    numerator=2*keras.backend.sum(y_true*y_pred)+0.0001
    denominator=keras.backend.sum(y_true**2)+keras.backend.sum(y_pred**2)+0.0001
    return 1-numerator/denominator/2
batch_size=32
train_data=id_data(batch_size=batch_size)
val_data=id_data(image_set='test',batch_size=batch_size)
# steps = math.ceil(len(glob.glob('data/mask/'+ '*.png')) / batch_size)
steps=12000/batch_size
optim=keras.optimizers.Adam()
a,b=next(iter(train_data))
model=model_initialize()
model.compile(optimizer=optim,loss='binary_crossentropy')
history=model.fit_generator(train_data,steps_per_epoch=steps,epochs=20,use_multiprocessing=True, verbose=2,workers=4,)
                            # validation_data=val_data,validation_steps=10,)
#                             callbacks=[keras.callbacks.EarlyStopping(monitor='val_iou',mode='max', patience=10, min_delta=0.01),
#                                        keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_iou', mode='max', verbose=1)])
keras.models.save_model(model,'id_dm05.h5',include_optimizer=False)

plt.plot(history.history['loss'], label='train')
plt.legend()
plt.show()

def picture(pre_numpy,img_numpy,mask_numpy):
    #pre_numpy has shape num,height,width,channel
    voc_colormap=np.array([[0, 0, 0], [245,222,179]])
    num=len(img_numpy)
    mean,std=np.array((0.485, 0.456, 0.406)),np.array((0.229, 0.224, 0.225))
    img=img_numpy*std+mean
    img=img.squeeze()
    mask=voc_colormap[mask_numpy.squeeze().astype(int)]
    tar=voc_colormap[pre_numpy.squeeze()]
    tmp=np.concatenate((img,tar,mask),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(3,num,i)
        plt.imshow(j)
    # plt.savefig('true_weight_alpha=05')
    plt.show()
def test(model,thread=0.5):
    img=[]
    pred=[]
    mask=[]
    l5_list=[]
    for image,mask_img in val_data:
        l5=model.predict(image)
        label=(l5>thread).squeeze().astype(int)
        l5_list.append(l5)
        pred.append(label)
        img.append(image)
        mask.append(mask_img)
    return np.concatenate(img,axis=0),np.concatenate(pred,axis=0),np.concatenate(mask,axis=0),l5_list
def thread(l5,mask):
    best_iou=[0,0]
    best_t=0
    best_label=0
    for i in np.arange(0.1,0.9,0.01):
        label=(l5>i).astype(int)
        ap,iou,hist,tmp=label_acc_score(mask,label,2)
        if iou[1]>best_iou[1]:
            best_iou=iou
            best_t=i
            best_label=label
    return  best_iou,best_t,best_label


model=keras.models.load_model('id_dm05.h5')
img,pred,mask,l=test(model)
ap,iou,hist,tmp=label_acc_score(mask,pred,2)
# iou,thread,pred=thread(np.concatenate(l,axis=0),mask)
picture(pred[0:4],img[0:4],mask[:4])

# with open('layer','w') as f:
#     for j,i in enumerate( model.layers):
#         f.writelines(str(j)+' '+i.name+"      "+str(len(i.get_weights()))+'\n')
