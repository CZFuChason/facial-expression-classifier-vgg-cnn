import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
slim = tf.contrib.slim


def inference(inputs):
    #Use Pretrained Base Model
    with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
          net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
          net = slim.max_pool2d(net, [2, 2], scope='pool1')
          net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
          net = slim.max_pool2d(net, [2, 2], scope='pool2')
          net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
          net = slim.max_pool2d(net, [2, 2], scope='pool3')
          net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
          net = slim.max_pool2d(net, [2, 2], scope='pool4')
          net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
          net = slim.max_pool2d(net, [2, 2], scope='pool5')
     # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
