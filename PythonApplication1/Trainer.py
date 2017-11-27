import cv2
import numpy as np
import tensorflow as tf
import random
from matplotlib import pyplot as plt



"""

Graph Building

"""


# Placeholders
X = tf.placeholder(tf.float32,[None,480,640])
X_re = tf.reshape(X,[-1,480,640,1])
Y = tf.placeholder(tf.float32,[None,3])

# Initial Pooling layer
pool1 = tf.nn.max_pool(X_re,[1,4,4,1],[1,4,4,1],padding='SAME')

# Convolutional layer
WC1 = tf.Variable(tf.truncated_normal(shape = [3,3,1,2],stddev=0.1), name='WC1')
BC1 = tf.Variable(tf.constant(0.1,shape = [2]),name='BC1')
conv1 = tf.nn.relu(tf.nn.conv2d(pool1, WC1, strides=[1,2,2,1], padding='SAME') + BC1)

WC2 = tf.Variable(tf.truncated_normal(shape = [5,5,2,4],stddev=0.1),name='WC2')
BC2 = tf.Variable(tf.constant(0.1,shape = [4]),name='BC2')
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, WC2, strides=[1,2,2,1], padding='SAME') + BC2)

# Fully connected layer
flatten = tf.reshape(conv2,[-1,4800])

WF1 = tf.Variable(tf.truncated_normal(shape = [4800,15], stddev=0.1), name = 'WF1')
BF1 = tf.Variable(tf.constant(0.1, shape = [15]), name = 'BF1')
fc1 = tf.nn.relu(tf.matmul(flatten,WF1)+BF1)
WF2 = tf.Variable(tf.truncated_normal(shape = [15,3], stddev = 0.1), name = 'WF2')
BF2 = tf.Variable(tf.constant(0.1, shape = [3]), name = 'BF2')
semi_result = tf.matmul(fc1,WF2)+BF2

# Result
result = tf.nn.softmax(semi_result)

# Evaluation
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=semi_result))
comp_pred = tf.equal(tf.arg_max(result,1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(comp_pred, dtype=tf.float32))

# Training
training_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



"""

Data aquisition

"""

print("Loading training data...")
input_set = []
validation_set = []

for i in range(1,701):
    input_set.append((cv2.imread("D:/RPSDATAv2/rock (%d).bmp"%i,0), [1,0,0]))
    input_set.append((cv2.imread("D:/RPSDATAv2/paper (%d).bmp"%i,0), [0,1,0]))
    input_set.append((cv2.imread("D:/RPSDATAv2/scissors (%d).bmp"%i,0), [0,0,1]))
    if i%100 == 0:
        print("Loaded training data : (%d/700)"%i)

for i in range(701,801):
    validation_set.append((cv2.imread("D:/RPSDATAv2/rock (%d).bmp"%i,0), [1,0,0]))
    validation_set.append((cv2.imread("D:/RPSDATAv2/paper (%d).bmp"%i,0), [0,1,0]))
    validation_set.append((cv2.imread("D:/RPSDATAv2/scissors (%d).bmp"%i,0), [0,0,1]))
    if i%100 == 0:
        print("Loaded validation data : (%d/700)"%i)

print("Data loading done")



"""

Training

"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saving_params = ['WC1','WC2','BC1','BC2','WF1','WF2','BF1','BF2']
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=10)

    cost_set = []
    accu_set = []
    for step in range(3001):
        in_X = []
        in_Y = []
        val_X = []
        val_Y = []
        # Data load
        for i in range(32):
            index = random.randint(0,2099)
            in_X.append(input_set[index][0])
            in_Y.append(input_set[index][1])
            index = random.randint(0,299)
            val_X.append(validation_set[index][0])
            val_Y.append(validation_set[index][1])

        # Validation
        sess.run(training_step, feed_dict={X:in_X,Y:in_Y})
        accu,cost = sess.run([accuracy,cross_entropy],feed_dict = {X:val_X,Y:val_Y})
        accu_set.append(accu); cost_set.append(cost)
        if step % 20 == 0:
            print("Step=%d, Accuracy=%f, Cost=%f"%(step,accu,cost))
            if step%100 == 0:
                print("Saving model...")
                saver.save(sess,'./save/model',global_step=step)
                print("Saving done")
                if step%1000 == 0 and step > 1:
                    plt.figure()
                    plt.plot(accu_set)
                    plt.show()
                    plt.figure()
                    plt.plot(cost_set)
                    plt.show()
                    accu_set.clear()
                    cost_set.clear()



