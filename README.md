# Semantic Segmentation using Advanced Deep Learning

### Overview


In this project the goal is to use semantic segmentation to label pixels of road images using a Fully Convolutional Network (FCN) based on the VGG-16 image classifier architecture.

![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/final_output.png?raw=true)

## Code description
The project consist of the following files:
* [`main.py`](./main.py) - holds most of the code. The [`run`](./main.py#L144) method downloads the pretrained VGG-16 model, then the [`load_vgg`](./main.py#L20) method extract the input layers of the VGG-16. The [`layers`](./main.py#L48) method adding decoder layers for FCN model: 1x1 Convolutions, Upsampling and Skip layers. After the network structure is set, the [`optimize`](./main.py#L88) method is called to build the TensorFLow loss and optimizer operations. The method [`train_nn`](./main.py#L114) is then ready to train the neural network (and print out the loss during training) using the FCN architecture and the optimizer we set. After the training, the method [`save_inference_samples`](./helper.py#L128) is running the NN on test images and save them to disk.
* [`project_tests.py`](./project_tests.py) - perform tests for the main methods of the project
* [`helper.py`](./helper.py) - for actions like downloading the pretrained vgg model, function to create batches of training data and Generate test output

## Architecture and Training

The project is based on the `Fully Convolutional Networks for Semantic Segmentation` paper (attached). Starting with the VGG-16 model, adding 1x1 convolutions on layers 3, 4 and 7, using L2 regularizer for each layer, adding transposed convolution layers for Upsampling, and skip connection layers. The final transpose layer for the output to match the input size was set with num_classes=2 (road or not).

![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/model_architecture.gif?raw=true)

I used Adam optimizer with learning rate of 0.001, and cross-entropy as the loss function.

For the training I added regularization dropout with keep probability = 0.5.

The dataset was trained several times, each time with different epoch (10, 25, 40, 50) and batch size (1, 5, 10).

The Final Hyperparameters with the best results:
* Epochs - 50
* Batch size - 5
* Learning rate - 0.001

Training was done on an Amazon GPU Spot instance type 'g3.4xlarge' and was running for ~32 minutes on the dataset.

The final loss value was 0.0572, and the project labels most pixels of roads close to the best solution.

## Sample images

Here are some test images from models that were trained with different Epochs. It's nice to observe how the segmentation improves over time.

10 Epochs     |  25 Epochs  |  40 Epochs  | 50 Epochs - Best results
:----------------:|:----------------:|:-----------------:|:-----------------:
![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/10_5/umm_000015.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/25_5/umm_000015.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/40_5/umm_000015.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/50_5/umm_000015.png?raw=true)
![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/25_5/uu_000005.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/40_5/uu_000005.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/50_5/uu_000005.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/10_5/uu_000005.png?raw=true)
![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/10_5/uu_000082.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/50_5/uu_000082.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/40_5/uu_000082.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/25_5/uu_000082.png?raw=true)
![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/50_5/uu_000095.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/25_5/uu_000095.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/40_5/uu_000095.png?raw=true)  |  ![]( https://github.com/shmulik-willinger/semantic_segmentation/blob/master/readme_img/10_5/uu_000095.png?raw=true)

 Visualizing this VGG16 model using **Tensorboard** can be extremely useful. Below I provided with a snippet to convert .pb file into TF summary. After converting it, we can run *tensorboard --logdir=.* in the same directory to start Tensorboard and visualize the graph in the browser
```
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.Session() as sess:     
  model_filename ='saved_model.pb'     
  with gfile.FastGFile(model_filename, 'rb') as f:        
    data = compat.as_bytes(f.read())         
    sm = saved_model_pb2.SavedModel()        
    sm.ParseFromString(data)        
    g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)

LOGDIR='.'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
```

Prerequisites and Executing
---

This project was built with python and requires the following dependencies:

 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

#### Dataset
For this project I used the [Kitti Road dataset](http://kitti.is.tue.mpg.de/kitti/data_road.zip) for training and testing. The dataset consist of images with size (160, 576). To use it you can extract the dataset in the `data` folder, this will create the folder `data_road` with all the training and test images.

#### Run
Run the following command to run the project:
```
python main.py
```
