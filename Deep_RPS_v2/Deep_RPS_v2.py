import cv2
import tensorflow as tf
import time

"""
constants
"""
NUM_CAM = 1


cap = cv2.VideoCapture(NUM_CAM)

# Build network model
X = tf.placeholder(tf.float32,[None,480,640,3])
X_re = tf.image.rgb_to_grayscale(X)

# Initial Pooling layer
pool1 = tf.nn.max_pool(X_re,[1,4,4,1],[1,4,4,1],padding='SAME')

# Convolutional layer
WC1 = tf.Variable(tf.truncated_normal(shape = [3,3,1,2],stddev=0.1), name='WC1')
BC1 = tf.Variable(tf.constant(0.1,shape = [2]),name='BC1')
conv1 = tf.nn.relu(tf.nn.conv2d(pool1, WC1, strides=[1,2,2,1], padding='SAME') + BC1)
conv1_drop = tf.nn.dropout(conv1,0.5)

WC2 = tf.Variable(tf.truncated_normal(shape = [5,5,2,4],stddev=0.1),name='WC2')
BC2 = tf.Variable(tf.constant(0.1,shape = [4]),name='BC2')
conv2 = tf.nn.relu(tf.nn.conv2d(conv1_drop, WC2, strides=[1,2,2,1], padding='SAME') + BC2)

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

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,'./save/model-500')

while True:
    if cv2.waitKey(200)>0 : break;
    _,frame = cap.read();
    res = sess.run(result, feed_dict={X:[frame]})
    print(res)
    cv2.imshow("test",frame)