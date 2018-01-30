import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input, keep, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    kernel_reg = tf.contrib.layers.l2_regularizer(1e-3)
    kernel_init = tf.random_normal_initializer(stddev=0.01)
    #kernel_init = tf.truncated_normal_initializer(stddev = 0.01)

    # 1x1 convolutions on layers 3, 4 and 7, using L2 regularizer
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, padding='same',
                                           kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, padding='same',
                                           kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, padding='same',
                                           kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)

    # upsampling - adding transposed convolution layers, and adding skip connection layers
    deconv_1 = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, kernel_size=4, strides=2, padding='same',
                                       kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)
    skip_1 = tf.add(deconv_1, conv_1x1_layer4)

    deconv_2 = tf.layers.conv2d_transpose(skip_1, num_classes, kernel_size=4, strides=2, padding='same',
                                       kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)
    skip_2 = tf.add(deconv_2, conv_1x1_layer3)

    # transpose layer for the output to match the input size
    deconv_3 = tf.layers.conv2d_transpose(skip_2, num_classes, kernel_size=16, strides=8, padding='same',
                                       kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)

    return deconv_3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # 2D tensor representing pixels (rows) and classes (columns)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # output labels
    labels = tf.reshape(correct_label, (-1, num_classes))
    # Adam optimizer with learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # cross-entropy as the loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())

    print("Training for {} epochs started".format(epochs))
    for epoch in range(epochs):
        #print("Epoch {}".format(epoch + 1))
        for images, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: images, correct_label: label, keep_prob: 0.5,
                                          learning_rate: 0.0009})
            print("Epoch {}".format(epoch + 1), " loss: {:.4f}".format(loss))
    print("Training for {} epochs finished successfully".format(epochs))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    #data_dir = './data'
    data_dir = "C:/Users/swillin/PycharmProjects/nano/CarND-Semantic-Segmentation-master"
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build the NN using load_vgg and layers
        print("Loading pretrained VGG model ...")
        input_image, keep_prob, layer3_out, layer_4_out, layer7_out = load_vgg(sess, vgg_path)
        print("Adding decoder layers for FCN model ...")
        output_layer = layers(layer3_out, layer_4_out, layer7_out, num_classes)

        # Optimize function
        print("Creating loss and optimizer ...")
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        print("Training the network ...")
        epochs = 50
        batch_size = 5
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # Save inference data
        print("Saving inference data ...")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
