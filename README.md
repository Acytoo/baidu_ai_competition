## 2018 Summer Internship II

### Goal
Store classfication.

### Dataset
[StoreTag](http://aistudio.baidu.com/aistudio/#/datasetDetail/274), or you can find it [here](https://drive.google.com/file/d/15EuZAkcaq5mkBUnq0cR5x33VTPINFQ1e/view?usp=sharing).

### Steps

- 0: Split train and test.

- 1: Building an ssd([more about ssd](http://arxiv.org/abs/1512.02325)) detector in paddlepaddle, train the neural network with train dataset. With special thanks to this [blog](https://blog.csdn.net/qq_33200967/article/details/79126830), we can fix an agonizing bug.

- 2: Building a ResNet to classify the tag area obtained in step1, this part of job is done by my teammate [Xu](https://github.com/LunHui123).

### Attentions:

* To run the ssd net, you need gpus, we have tested on Debian9, Cuda8, cudnn7, python2.7.15(```pip2 install paddlepaddle-gpu==0.14.0.post87```) and kali-rolling, Cuda9.1, cudnn7, python2.7.15(```pip2 install paddlepaddle-gpu```)

* To start training, you need a pre-trained [model](http://paddlepaddle.bj.bcebos.com/model_zoo/detection/ssd_model/vgg_model.tar.gz), or there will be float errors. Modify the pre-trained model as what your net is, here you need to delete all the files with 'mbox'(```rm *mbox*```).

### Screenshots:
![image](https://acytoo.github.io/HPSRC/2018Internship/2018internship0.png)
![image](https://acytoo.github.io/HPSRC/2018Internship/2018internship1.png)
![image](https://acytoo.github.io/HPSRC/2018Internship/2018internship2.png)
![image](https://acytoo.github.io/HPSRC/2018Internship/2018internship3.png)

### [part I](https://github.com/Acytoo/2018SummerInternshipI)
