import numpy as np
import tensorflow as tf


class ModelFeeddict(object):
    def __init__(self, num_epochs=10):
        self.num_epochs = num_epochs
        self.train_data, self.train_labels, self.eval_data, self.eval_labels = self.load_data()

        self.images = tf.placeholder(tf.float32, shape=[None, 28 * 28])
        self.input = tf.reshape(tensor=self.images, shape=[-1, 28, 28, 1])
        self.labels = tf.placeholder(tf.int32, shape=[None, ])

        conv1 = tf.layers.conv2d(self.input, 32, [5, 5], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(pool1, 64, [5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])

        dense = tf.layers.dense(inputs=pool2, units=1024, activation=tf.nn.relu)

        logits = tf.layers.dense(inputs=dense, units=10)

        self.class_prediction = tf.argmax(logits, axis=1)
        self.prediction_probabilites = tf.nn.softmax(logits)

        self.loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.labels)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        self.accuracy = tf.metrics.accuracy(labels=self.labels, predictions=self.class_prediction)

        self.session = tf.Session()

    def load_data(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        return train_data, train_labels, eval_data, eval_labels

    def train(self, batch_size=128, max_batches=10000):
        self.session.run(tf.global_variables_initializer())

        num_batches = int(np.ceil(self.train_data.shape[0] / batch_size))

        processed_batches = 0
        for i in range(self.num_epochs):
            for j in range(num_batches):
                batch = self.train_data[j*batch_size:(j+1)*batch_size]
                batch_labels = self.train_labels[j*batch_size:(j+1)*batch_size]
                self.session.run(self.train_op, feed_dict={self.images: batch, self.labels: batch_labels})

                processed_batches += 1
            if processed_batches >= max_batches:
                break


if __name__ == '__main__':
    m = ModelFeeddict()
    m.train()
