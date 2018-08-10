import tensorflow as tf

def creat_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def creat_biases(size):
    return tf.Variable(tf.constant(0.01, shape=size))

def creat_conv_layer(input, num_channels, conv_filter_size, num_filters):
    weights = creat_weights(shape=[conv_filter_size, conv_filter_size, num_channels, num_filters])
    biases = creat_biases(size=[num_filters])

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')
    layer = layer+biases

    layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    layer = tf.nn.relu(layer)

    return layer

def creat_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer,[-1, num_features])

    return layer

def creat_fc_layer(input, num_inputs, num_outputs, use_relu):
    weights = creat_weights(shape=[num_inputs,num_outputs])
    biases = creat_biases(size=[num_outputs])

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def creat_softmax_layer(input, num_inputs, num_outputs):
    weights = weights = creat_weights(shape=[num_inputs,num_outputs])
    biases = creat_biases(size=[num_outputs])
    layer = tf.matmul(input, weights) + biases
    return layer


#define CNN graph parameters
filter_size_conv1 = 3
num_filters_conv1 = 16

filter_size_conv2 = 3
num_filters_conv2 = 32

fc1_layer_size = 2048
fc2_layer_size = 512

num_channels = 1
num_classes = 7

def inference(input, reuse):
    layer_conv1 = creat_conv_layer(input=input, num_channels=num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1)

    layre_conv2 = creat_conv_layer(input=layer_conv1, num_channels=num_filters_conv1, conv_filter_size=filter_size_conv2,num_filters=num_filters_conv2)

    layer_flat = creat_flatten_layer(layre_conv2)

    layer_fc1 = creat_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                    num_outputs=fc1_layer_size,
                                    use_relu=True)

    if not reuse:
        layer_fc1 = tf.nn.dropout(layer_fc1, keep_prob=0.5)

    layer_fc2 = creat_fc_layer(input=layer_fc1, num_inputs=fc1_layer_size,
                                    num_outputs=fc2_layer_size,
                                    use_relu=True)

    if not reuse:
        layer_fc2 = tf.nn.dropout(layer_fc2, keep_prob=0.5)


    y_pred = creat_softmax_layer(layer_fc2, fc2_layer_size, num_classes)
    #y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
    #cls_result = tf.argmax(y_pred, axis=1, name='cls_result')
    #cls_result = tf.cast(cls_result, tf.float32)

    #return cls_result
    return y_pred


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='entropy_per_example')

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name=scope.name)
        #loss = tf.add_n(tf.get_collection("losses")) + cross_entropy_mean
        tf.summary.scalar(scope.name + '/loss', cross_entropy_mean)
    #return loss
    return cross_entropy_mean


def training(loss, learning_rate):
    with tf.variable_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        train_op = optimizer.minimize(loss, global_step=global_step, name=scope.name)
    return train_op


def evaluation(logits, labels, size):
    # with tf.variable_scope('accuracy') as scope:
    #     # correct = tf.nn.in_top_k(tf.cast(logits[i], tf.float32) for i in range(size), tf.cast(labels[i], tf.float32) for i in range(size), 1)
    #     # correct = tf.cast(correct, tf.float16)
    #     correct = [ tf.equal(tf.cast(logits[i], tf.float32), tf.cast(labels[i], tf.float32)) for i in range(size)]
    #     accuracy = tf.reduce_mean(correct)
    #     tf.summary.scalar(scope.name + '/accuracy', accuracy)
    # return accuracy
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
