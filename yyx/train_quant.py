import os
import tensorflow as tf
# from tensorflow.core.protobuf import saver_pb2
import driving_data
import models.model_mb3x3_standard as model

LOGDIR = './save'
g = tf.get_default_graph()
tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=0)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
# checkpoint_path = os.path.join(LOGDIR, "model_mb3x3_s4.ckpt")
# saver.restore(sess, checkpoint_path)


train_vars = tf.trainable_variables()
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))
regular = tf.add_n([tf.nn.l2_loss(v) for v in train_vars])
global_step = tf.Variable(tf.constant(0))
L2NormConst = tf.train.exponential_decay(0.004, global_step, 3000, 0.5, staircase=True)
learning_rate = tf.train.exponential_decay(0.0002, global_step, 450, 0.85, staircase=True)
loss += tf.multiply(regular, L2NormConst)
# 这两句话用于更新moving mean和variance
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess.run(tf.global_variables_initializer())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
tf.summary.scalar('L2Norm', L2NormConst)
tf.summary.scalar('learning_rate', learning_rate)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 15
batch_size = 150
sum_step = 0
# train over the dataset about 30 times
for epoch in range(epochs):
    epoch_step = int(driving_data.num_images / batch_size)
    for i in range(epoch_step):
        sum_step += 1
        xs, ys = driving_data.LoadTrainBatch(batch_size)
        train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8, global_step: sum_step})
        if i % 10 == 0:
            xs, ys = driving_data.LoadValBatch(batch_size)
            loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0, global_step: sum_step})
            print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * epoch_step + i, loss_value))
        # write logs at every iteration
        summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0, global_step: sum_step})
        summary_writer.add_summary(summary, epoch * epoch_step + i)

    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if sum_step < 4000:
        checkpoint_path = os.path.join(LOGDIR, "model_mb3x3s_float.ckpt")
    else:
        checkpoint_path = os.path.join(LOGDIR, "model_mb3x3s_quant.ckpt")
    filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)

with open('./save/eval_freeze_graph.pb', 'w') as f:
    f.write(str(g.as_graph_def()))

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")

