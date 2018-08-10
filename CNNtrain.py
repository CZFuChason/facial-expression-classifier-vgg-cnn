import tensorflow as tf
import loader
import CNN
import numpy as np
import os

train_dir = "./fer2013/train/"
logs_train_dir = "./train"
logs_validation_dir = "./fer2013/val/"


N_CLASSES = 7
IMG_W = 48
IMG_H = 48
TRAIN_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 100
CAPACITY = 256
MAX_STEP = 50000
LEARNING_RATE = 0.0001

def train():

    train_data, train_label = loader.get_file(file_dir=train_dir)
    train_batch, train_label_batch = loader.get_batch(train_data, train_label,
                                                      IMG_W, IMG_H,
                                                      TRAIN_BATCH_SIZE,CAPACITY)

    validation, validation_label = loader.get_file(file_dir=logs_validation_dir)
    validation_batch, validation_label_batch = loader.get_batch(validation, validation_label,
                                                                IMG_W, IMG_H,
                                                                VALIDATION_BATCH_SIZE, CAPACITY)

    #tf.Session().run(train_label_batch)
    #train_label_batch = tf.reshape(train_label_batch, [TRAIN_BATCH_SIZE,-1])

    train_logits_op = CNN.inference(input=train_batch, reuse=False)
    validation_logits_op = CNN.inference(input=validation_batch, reuse=False)
    train_losses_op = CNN.losses(logits=train_logits_op, labels=train_label_batch)
    validation_losses_op = CNN.losses(logits=validation_logits_op, labels=validation_label_batch)

    train_op = CNN.training(train_losses_op, learning_rate=LEARNING_RATE)

    train_accuracy_op = CNN.evaluation(logits=train_logits_op, labels=train_label_batch, size=TRAIN_BATCH_SIZE)

    validation_accuracy_op = CNN.evaluation(logits=validation_logits_op, labels=validation_label_batch, size=VALIDATION_BATCH_SIZE)

    #train_cross_entropy_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_op, labels=train_label_batch))
    # train_accuracy = [ tf.equal(tf.cast(train_op[i], tf.float32), tf.cast(train_label_batch[i], tf.float32)) for i in range(TRAIN_BATCH_SIZE)]
    # train_accuracy = tf.reduce_mean(tf.cast(train_accuracy, tf.float32))
    #
    # #validation_cross_entropy_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=validation_label_batch,logits=validation_op))
    # val_accuracy = [ tf.equal(tf.cast(train_op[i], tf.float32), tf.cast(validation_label_batch[i], tf.float32)) for i in range(VALIDATION_BATCH_SIZE)]
    # val_accuracy = tf.reduce_mean(tf.cast(val_accuracy, tf.float32))
    #
    #
    # optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(train_accuracy)
    # train_correct_prediction = tf.equal(train_op, train_label_batch)
    # train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

    # validation_correct_prediction = tf.equal(validation_op, validation_label_batch)
    # validation_accuracy = tf.reduce_mean(tf.cast(validation_correct_prediction, tf.float32))

    #tf.summary.scalar('train_loss', train_cross_entropy_op)
    #tf.summary.scalar('train_accuracy', train_accuracy)
    #tf.summary.scalar('val_loss', validation_correct_prediction)
    #tf.summary.scalar('val_accuracy', validation_accuracy)

    merge_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("./train/summary")
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in range(MAX_STEP):
                if coord.should_stop():
                    break
                #loss = sess.run(train_cross_entropy_op)
                _, train_loss, train_accuracy = sess.run([train_op, train_losses_op, train_accuracy_op])
                if step % 100 ==0:
                    #print('Step %d, training accuracy %.2f, loss %.2f'%(step, acc*100.0, loss))
                    print('Step %d, train loss = %.2f, train accuracy = %.2f' % (step, train_loss, train_accuracy * 100.0))
                    summery_str = sess.run(merge_summary)
                    train_writer.add_summary(summery_str,step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'EMOTION_CNN.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

                if step % 500 == 0 or (step + 1) == MAX_STEP:
                    #val_loss, val_accuracy = sess.run([validation_cross_entropy_op, validation_accuracy])
                    val_loss, val_accuracy = sess.run([validation_losses_op, validation_accuracy_op])
                    print('** step %d, val loss = %.2f, val accuracy = %.2f' % (step, val_loss, val_accuracy * 100.0))
                    summery_str = sess.run(merge_summary)
                    train_writer.add_summary(summery_str,step)

        except tf.errors.OutOfRangeError:
            print("Done training -- epoch limit reached")

        finally:
            coord.request_stop()


if __name__ == '__main__':
    train()
