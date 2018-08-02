import paddle
import paddle.fluid as fluid
from train_res import *
import cv2



def inference_program():
    data = fluid.layers.data(name='img', shape=[3,32,32], dtype='float32')
    predict = ResNet50().net(data)
    return predict


def total_infer(img):
    inferencer = fluid.Inferencer(
        infer_func=inference_program, param_path='../data/dataset/inference.model', place=fluid.CUDAPlace(0))

    result = inferencer.infer({'img':img})

def load_image(img):
    #img=Image.open(imgPath)
    #img=img.crop((int(coner[0]),int(coner[1]),int(coner[2]),int(coner[3])))
    #img=img.resize((32,32),Image.ANTIALIAS)
    #if img.mode != 'RGB':
    #    img=img.convert('RGB')
    imgMat=img.astype('float32').transpose((2,0,1))/255.0
    #imgMat=imgMat.flatten().reshape(1,3,32,32) 
    
    imgMat=np.expand_dims(imgMat,axis=0)

    return imgMat


if __name__ == '__main__':
    res_path = '../data/infer.res'
    res_file = open(res_path, 'r')
    for each_line in res_file.readlines():
        parts = each_line.replace('\t', ' ').split(' ')
        print(parts)
        xmin = float(parts[3])
        ymin = float(parts[4])
        xmax = float(parts[5])
        ymax = float(parts[6])
        img = cv2.imread('../data/'+parts[0])
        img_tag = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        imgMat = cv2.cvtColor(img_tag, cv2.COLOR_BGR2RGB)
        imgMat = cv2.resize(imgMat, (32,32))
        imgMat = img.astype('float32').transpose((2,0,1))/255.0
        imgMat = np.expand_dims(imgMat,axis=0)

        res = total_infer(imgMat)
        cv2.imshow('ori', img_tag)
        print(res)


