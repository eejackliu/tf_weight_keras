import concurrent.futures
import time
import  glob
import cv2 as cv
import os
import numpy as np
from PIL import Image as image
import zipfile
import json
# pth='/media/llm/WB/data/tmp'
# q=glob.glob(pth)
def cvimg(filename):
    try:
        a=cv.imread(filename)
        # b=cv.rotate(a,cv.ROTATE_90_CLOCKWISE)
        b=cv.resize(a,(224,224))
        cv.imwrite('/media/llm/WB/data/test/'+os.path.basename(filename),b)
    except:
        print(filename)
def pilimg(filename):
    a=image.open(filename)
    a.resize((224,224)).save('/media/llm/WB/data/test/'+os.path.basename(filename))
# filename=glob.glob('/media/llm/WB/data/tmp/**/*.jpg',recursive=True)
# filename=glob.glob('/media/llm/WB/data/sm/formybg/*.jpg')

# start_time=time.time()
# with concurrent.futures.ProcessPoolExecutor(max_workers=4) as excutor:
#     futures=[excutor.submit(pilimg,item) for item in filename]
# stop_time=time.time()
# print('piltime      {}\n'.format(stop_time-start_time))
#
#
# start_time=time.time()
# with concurrent.futures.ProcessPoolExecutor(max_workers=4) as excutor:
#     futures=[excutor.submit(cvimg,item) for item in filename]
# stop_time=time.time()
# print('cvtime      {}\n'.format(stop_time-start_time))
# def jspic(filename):

def pic(filename):
    # filename='/media/llm/D08E2AF4DD65FFFC/data/01_alb_id/ground_truth/CA/CA01_01.json'
    f=open(filename)
    js=json.load(f)
    name=filename.split('/')[1:]
    name[-3]='images'
    name[-1]=name[-1][:-4]+'tif'
    img_name='/media/llm/D08E2AF4DD65FFFC/data/img/'+'-'.join(name[-4:])
    # mask_name=img_name[:-3]+'jpg'
    mask_name='/media/llm/D08E2AF4DD65FFFC/data/mask/'+'-'.join(name[-4:])[:-3]+'jpg'
    name=['/'+i for i in name]
    a=cv.imread(''.join(name))
    if a.shape[0]/a.shape[1]>1:
        shape=(184,320)
    else:
        shape=(320,184)

    # b=cv.resize(a,tuple([int(i/6) for i in a.shape[1::-1]]))
    b=cv.resize(a,shape)
    cv.imwrite(img_name,b)
    img=np.zeros(a.shape,np.uint8)
    mask=cv.fillConvexPoly(img,np.array(js['quad']),[255,255,255])
    # mask_b=cv.resize(mask,tuple([int(i/6) for i in mask.shape[1::-1]]))
    mask_b=cv.resize(mask,shape)
    cv.imwrite(mask_name,mask_b)

def test(filename):
    a=cv.imread(filename)
    if a.shape[0]/a.shape[1]<1:
        print(filename)

# filename=glob.glob('/media/llm/D08E2AF4DD65FFFC/data/**/*.json',recursive=True)
# # filename=glob.glob('/media/llm/D08E2AF4DD65FFFC/data/mask/*.jpg')
# start_time=time.time()
# with concurrent.futures.ProcessPoolExecutor(max_workers=4) as excutor:
#     futures=[excutor.submit(pic,item) for item in filename]
# stop_time=time.time()
# print('piltime      {}\n'.format(stop_time-start_time))
