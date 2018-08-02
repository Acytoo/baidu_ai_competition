import math
from PIL import Image
import paddle
from paddle import fluid
from paddle import v2
import csv
import random
import os
import numpy as np
import sys

############## - Image Load- ################
def load_image(imgPath,coner):
    img=Image.open(imgPath)
    img=img.crop((int(coner[0]),int(coner[1]),int(coner[2]),int(coner[3])))
    img=img.resize((32,32),Image.ANTIALIAS)
    if img.mode != 'RGB':
        img=img.convert('RGB')
    imgMat=np.array(img).astype('float32').transpose((2,0,1))/255.0
    #imgMat=imgMat.flatten().reshape(1,3,32,32) 
    
    imgMat=np.expand_dims(imgMat,axis=0)

    return imgMat

def load_image_gray(imgPath,coner):
    img=Image.open(imgPath).convert("L")
    img=img.crop((int(coner[0]),int(coner[1]),int(coner[2]),int(coner[3])))
    img=img.resize((28,28),Image.ANTIALIAS)
    imgMat=np.array(img).reshape(1,1,32,32).astype(np.float32)
    imgMat=imgMat/255.0*2.0-1.0
    return imgMat.flatten()
 
def get_img_reader(path,imgList):
    imgDir=os.path.join(path,"train")
    def reader():
        for item in imgList:
            imgPath=os.path.join(imgDir,item[0])
            imgMat=load_image(imgPath,item[2:])
            
            t=int(item[1])
            yield imgMat,t
    return reader

############### - Image Load End- #############

############### - Google Net - ###############

class GoogleNet():
    def __init__(self):
        self.params = {
    "input_size": [3, 32, 32],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
        }
    }

    def conv_layer(self,
                   input,
                   num_filters,
                   filter_size,
                   stride=1,
                   groups=1,
                   act=None):
        channels = input.shape[1]
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv))
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) / 2,
            groups=groups,
            act=act,
            param_attr=param_attr,
            bias_attr=False)
        return conv

    def xavier(self, channels, filter_size):
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv))
        return param_attr

    def inception(self, name, input, channels, filter1, filter3R, filter3,
                  filter5R, filter5, proj):
        conv1 = self.conv_layer(
            input=input, num_filters=filter1, filter_size=1, stride=1, act=None)
        conv3r = self.conv_layer(
            input=input,
            num_filters=filter3R,
            filter_size=1,
            stride=1,
            act=None)
        conv3 = self.conv_layer(
            input=conv3r,
            num_filters=filter3,
            filter_size=3,
            stride=1,
            act=None)
        conv5r = self.conv_layer(
            input=input,
            num_filters=filter5R,
            filter_size=1,
            stride=1,
            act=None)
        conv5 = self.conv_layer(
            input=conv5r,
            num_filters=filter5,
            filter_size=5,
            stride=1,
            act=None)
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=1,
            pool_padding=1,
            pool_type='max')
        convprj = fluid.layers.conv2d(
            input=pool, filter_size=1, num_filters=proj, stride=1, padding=0)
        cat = fluid.layers.concat(input=[conv1, conv3, conv5, convprj], axis=1)
        cat = fluid.layers.relu(cat)
        return cat

    def net(self, input, class_dim=61):
        conv = self.conv_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act=None)
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)

        conv = self.conv_layer(
            input=pool, num_filters=64, filter_size=1, stride=1, act=None)
        conv = self.conv_layer(
            input=conv, num_filters=192, filter_size=3, stride=1, act=None)
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)

        ince3a = self.inception("ince3a", pool, 192, 64, 96, 128, 16, 32, 32)
        ince3b = self.inception("ince3b", ince3a, 256, 128, 128, 192, 32, 96,
                                64)
        pool3 = fluid.layers.pool2d(
            input=ince3b, pool_size=3, pool_type='max', pool_stride=2)

        ince4a = self.inception("ince4a", pool3, 480, 192, 96, 208, 16, 48, 64)
        ince4b = self.inception("ince4b", ince4a, 512, 160, 112, 224, 24, 64,
                                64)
        ince4c = self.inception("ince4c", ince4b, 512, 128, 128, 256, 24, 64,
                                64)
        ince4d = self.inception("ince4d", ince4c, 512, 112, 144, 288, 32, 64,
                                64)
        ince4e = self.inception("ince4e", ince4d, 528, 256, 160, 320, 32, 128,
                                128)
        pool4 = fluid.layers.pool2d(
            input=ince4e, pool_size=3, pool_type='max', pool_stride=2)

        ince5a = self.inception("ince5a", pool4, 832, 256, 160, 320, 32, 128,
                                128)
        ince5b = self.inception("ince5b", ince5a, 832, 384, 192, 384, 48, 128,
                                128)
        pool5 = fluid.layers.pool2d(
            input=ince5b, pool_size=7, pool_type='avg', pool_stride=7)
        dropout = fluid.layers.dropout(x=pool5, dropout_prob=0.4)
        out = fluid.layers.fc(input=dropout,
                              size=class_dim,
                              act='softmax',
                              param_attr=self.xavier(1024, 1))

        pool_o1 = fluid.layers.pool2d(
            input=ince4a, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o1 = self.conv_layer(
            input=pool_o1, num_filters=128, filter_size=1, stride=1, act=None)
        fc_o1 = fluid.layers.fc(input=conv_o1,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1))
        dropout_o1 = fluid.layers.dropout(x=fc_o1, dropout_prob=0.7)
        out1 = fluid.layers.fc(input=dropout_o1,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1))

        pool_o2 = fluid.layers.pool2d(
            input=ince4d, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o2 = self.conv_layer(
            input=pool_o2, num_filters=128, filter_size=1, stride=1, act=None)
        fc_o2 = fluid.layers.fc(input=conv_o2,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1))
        dropout_o2 = fluid.layers.dropout(x=fc_o2, dropout_prob=0.7)
        out2 = fluid.layers.fc(input=dropout_o2,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1))

        # last fc layer is "out"
        return out, out1, out2

############### - Google Net End - ###############


############### - Vgg bn drop - ###############

def vgg_bn_drop(data):
    
    img=data

    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
        input=ipt,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[num_filter] * groups,
        conv_filter_size=3,
        conv_act='relu',
        conv_with_batchnorm=True,
        conv_batchnorm_drop_rate=dropouts,
        pool_type='max')

    conv1 = conv_block(img, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=61, act='softmax')
    return predict


############### - Vgg bn drop End - ###############


############### - Convolutional neural network - ###############

def convolutional_neural_network(data):
    img=data

    conv_pool_1 = fluid.nets.simple_img_conv_pool(
                                           input=img,
                                            filter_size=5,
                                            num_filters=20,
                                            pool_size=2,
                                            pool_stride=2,
                                            act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
                                            input=conv_pool_1,
                                            filter_size=5,
                                            num_filters=50,
                                            pool_size=2,
                                            pool_stride=2,
                                            act="relu")
    prediction = fluid.layers.fc(input=conv_pool_2, size=61, act='softmax')
    return prediction

############### - Convolutional neural network End - ###############


############### - ResNet - ###############
def conv_bn_layer(input,
        ch_out,
        filter_size,
        stride,
        padding,
        act='relu',
        bias_attr=False):

    tmp = fluid.layers.conv2d(
    input=input,
    filter_size=filter_size,
    num_filters=ch_out,
    stride=stride,
    padding=padding,
    act=None,
    bias_attr=bias_attr)
    return fluid.layers.batch_norm(input=tmp, act=act)


def shortcut(input, ch_in, ch_out, stride):
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_in, ch_out, stride):
    tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
    short = shortcut(input, ch_in, ch_out, stride)
    return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')


def layer_warp(block_func, input, ch_in, ch_out, count, stride):
    tmp = block_func(input, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1)
    return tmp


def resnet(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(ipt, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
    input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    predict = fluid.layers.fc(input=pool, size=61, act='softmax')
    
    return predict


############### - ResNet End - ###############

############### - ResNet Class  - ###############

class ResNet():
    def __init__(self, layers=50):
        self.params = {
	    "input_size": [3, 32, 32],
	    "input_mean": [0.485, 0.456, 0.406],
	    "input_std": [0.229, 0.224, 0.225],
	    "learning_strategy": {
		"name": "piecewise_decay",
		"batch_size": 256,
		"epochs": [30, 60, 90],
		"steps": [0.1, 0.01, 0.001, 0.0001]
	    }
	} 
	self.layers = layers

    def net(self, input, class_dim=61):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1)

        pool = fluid.layers.pool2d(
            input=conv, pool_size=7, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=pool,
                              size=class_dim,
                              act='softmax',
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv,
                                                                        stdv)))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) / 2,
            groups=groups,
            act=None,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 4, filter_size=1, act=None)

        short = self.shortcut(input, num_filters * 4, stride)

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def ResNet50():
    model = ResNet(layers=50)
    return model


def ResNet101():
    model = ResNet(layers=101)
    return model


def ResNet152():
    model = ResNet(layers=152)
    return model
############### - ResNet Class End - ###############


def train_program():
    label=fluid.layers.data(name='label',shape=[1],dtype='int64')

    data=fluid.layers.data(name='img',shape=[3,32,32],dtype='float32')
    ''' 
    #google net
    net=GoogleNet()
    out1,out2,out3=net.net(data)
    
    cost1=fluid.layers.cross_entropy(input=out1,label=label)
    cost2=fluid.layers.cross_entropy(input=out2,label=label)
    cost3=fluid.layers.cross_entropy(input=out3,label=label)
    
    avg_cost1=fluid.layers.mean(cost1)
    avg_cost2=fluid.layers.mean(cost2)
    avg_cost3=fluid.layers.mean(cost3)
    
    avg_cost=avg_cost1+0.3*avg_cost2+0.3*avg_cost3
    acc=fluid.layers.accuracy(input=out1,label=label)
    #google net
    '''
    #predict=convolutional_neural_network(data)
    #predict=vgg_bn_drop(data)
    #predict=resnet(data)
    predict=ResNet50().net(data)

    print("calc cost")
    cost= fluid.layers.cross_entropy(input=predict,label=label)
    avg_cost=fluid.layers.mean(cost)
    print("calc accuracy") 
    acc = fluid.layers.accuracy(input=predict, label=label)

    return [avg_cost,acc]

def optimizer():
    return fluid.optimizer.Adam(learning_rate=0.001)

def event_handler():
    params_dirname=os.path.join(path,"inference.model")
    def event_handle(event):
        if isinstance(event,fluid.EndStepEvent):
            if event.step % 100 == 0:
                print("\nStep %d, Pass %d, Cost %f ,Acc %f" %(event.step, event.epoch, event.metrics[0],event.metrics[1]))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event,fluid.EndEpochEvent):
            result = trainer.test(reader=test_reader, feed_order=['img', 'label'])
            print('\nTest with Pass {0},Loss {0},Acc {2:2.2}'.format(event.epoch,result[0],result[1]))

            # save parameters
            if params_dirname is not None:
                trainer.save_params(params_dirname) 
          
    return event_handle



if __name__ == '__main__':
    print("start")

    path=r"../data/dataset"

    #img=Image.open(r"/home/train/023b5bb5c9ea15ce0082c2b9bd003af33a87b215.jpg")
    print("prepare data")

    csvreader=csv.reader(file(os.path.join(path,r"train.txt")))
    imgList=[]
    for row in csvreader:
        imgList.append(row)

    random.shuffle(imgList)

    trainList=imgList[len(imgList)//4:]
    testList=imgList[:len(imgList)//4]

    use_cuda=True
    place=fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    print("prepare trainer")

    trainer=fluid.Trainer(
            train_func=train_program,
            place=place,
            optimizer_func=optimizer)



    train_reader=paddle.batch(paddle.reader.shuffle(get_img_reader(path,trainList),buf_size=500),batch_size=64)
    test_reader=paddle.batch(paddle.reader.shuffle(get_img_reader(path,testList),buf_size=500),batch_size=64)

    #train_reader=paddle.batch(paddle.reader.shuffle(paddle.dataset.cifar.train10(),buf_size=500),batch_size=64)
    #test_reader=paddle.batch(paddle.reader.shuffle(paddle.dataset.cifar.test10(),buf_size=500),batch_size=64)

    print("start train")


    trainer.train(num_epochs=30,
                event_handler=event_handler(),
                reader=train_reader,
                feed_order=['img','label']
                )
