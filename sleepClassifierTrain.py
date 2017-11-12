from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from PIL import Image

PATH_TO_TEST_IMAGES_DIR = "D:\\tmp\\testImages"
TEST_IMAGES_PATH = [os.path.join(PATH_TO_TEST_IMAGES_DIR, "image{}.png".format(i)) for i in range(1,5)]
TRAIN_DATA = "D:\\tmp\\sleep_train_data_6k.npz"
#TRAIN_DATA = "D:\\tmp\\sleep_train_data_60k.npz"
TEST_DATA = "D:\\tmp\\sleep_test_data_50k.npz"

tf.logging.set_verbosity(tf.logging.INFO)
def cnn_model_fn(features,labels,mode):
        """model function for cnn."""
        # Input Layer
        input_layer=tf.reshape(features["x"], [-1, 128, 106, 1])
        print(features["x"])
        #Conv layer 1
        conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[25,25],
                padding="same",
                activation=tf.nn.relu)

        # Pooling layer 1
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[2,2],
                                        strides=2)

        # conv layer2
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[25,25],
                padding="same",
                activation=tf.nn.relu)

        #pooling layer2
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[2,2],
                                        strides=2)

         # conv layer3
        conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=64,
                kernel_size=[25,25],
                padding="same",
                activation=tf.nn.relu)

        #pooling layer3
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                        pool_size=[2,2],
                                        strides=2)


       

        # dense layer
        pool3_flat = tf.reshape(pool3,[-1,16*13*64])
        dense = tf.layers.dense(inputs=pool3_flat,units=1024,activation=tf.nn.relu)
        #dense2 = tf.layers.dense(inputs=dense1,units=512,activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense,
                                  rate=0.4,
                                  training = mode == tf.estimator.ModeKeys.TRAIN)

        # logits layer
        logits = tf.layers.dense(inputs=dropout,units=5)

        predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes" : tf.argmax(input=logits, axis=1),
                # softmax for logging and predict
                "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
                }

        if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # calculate loss ( for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=5)
        loss = tf.losses.softmax_cross_entropy(
                onehot_labels = onehot_labels, logits = logits)
        
        #Configure the training Op ( for TRAIN mode)

        if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
                #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                #optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
                train_op = optimizer.minimize(
                        loss=loss,
                        global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode,loss=loss, train_op=train_op)

        # Add evaluation metric (for EVAL mode)
        eval_metric_ops = {
                "accuracy":tf.metrics.accuracy(
                        labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
                mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)



def main(unused_argv):

        sleepData_train = np.load(TRAIN_DATA) 
        train_data = sleepData_train['image_data'].astype(dtype=np.float16)
        
        train_labels =  sleepData_train['labels'].astype(dtype=np.int32)
        sleepData_test = np.load(TEST_DATA)
        eval_data = sleepData_test['image_data'].astype(dtype=np.float16)
        eval_labels = sleepData_test['labels'].astype(dtype=np.int32)
        #eval_data = mnist.test.images
        #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        # Create the estimator
        mnist_classifier = tf.estimator.Estimator(
                model_fn=cnn_model_fn, model_dir="tmp/sleep_convnet_model")

        # set up logging for predictions
        tensors_to_log = {"probabilities":"softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=50)

        # train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x":train_data},
                y=train_labels,
                batch_size=20,
                num_epochs=None,
                shuffle=True)
        mnist_classifier.train(
                input_fn=train_input_fn,
                steps=300000,
                hooks=[logging_hook])


       #  Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x":eval_data},
                y=eval_labels,
                num_epochs=1,
                shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

if __name__ == "__main__":
	tf.app.run()
	