import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import re

#---------------------------------------------------------------------------------------
## reading the pgm file
file_no = 16

labels= []
images= []
for i in xrange(11,file_no+1):
	if i==14: continue
	nmbr = str(i)
	dirc = 'yaleB' + nmbr
	for j in xrange(1,9):
		if i==16: nmbr2 = '0'+str(j+1)
		else: nmbr2='0'+str(j)
		file_info = open(dirc+'/'+dirc+'_P'+nmbr2+'.info','r')
		label = np.zeros(5)

		for line in file_info.readlines():
			line = line.strip()
			with open(dirc+'/'+line, 'rb') as f:
			  buffer = f.read()
			try:
			  header, width, height, maxval = re.search(
			  b"(^P5\s(?:\s*#.*[\r\n])*"
			  b"(\d+)\s(?:\s*#.*[\r\n])*"
			  b"(\d+)\s(?:\s*#.*[\r\n])*"
			  b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
			except AttributeError:
			  raise ValueError("Not a raw PGM file: '%s'" % filename)
			#print line
			xh=np.frombuffer(buffer,
				   dtype='u1' if int(maxval) < 256 else byteorder+'u2',
				   count=int(width)*int(height),
				   offset=len(header)
				  ).reshape((int(height)*int(width)))
			if i<14:
				label[i-11]=1
			else:
				label[i-12]=1
			if xh.shape[0] == 32256:
				labels.append(label)
				images.append(xh)
			#print i, xh.shape
		

y_labels = np.asarray(labels)
x_images = np.asarray(images)/255.0

tt = y_labels.shape[0]

# shuffle the two
s = np.arange(tt)
np.random.shuffle(s)

x_images = x_images[s]
y_labels = y_labels[s]

file_info.close()

#----------------------------------------------------------------------------------------------------------


sess = tf.InteractiveSession()

# Functions for creating weights and biases
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Functions for convolution and pooling functions
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    
# Create placeholders nodes for images and label inputs
x = tf.placeholder(tf.float32, shape=[None, 168*192])
y_ = tf.placeholder(tf.float32, shape=[None, 5])


# y = (Wx +b)

# Input layer
x_image = tf.reshape(x, [-1,168,192,1]) 

# Conv layer 1 - 32x5x5
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
x_pool1 = max_pooling_2x2(x_conv1)

# Conv layer 2 - 64x5x5
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
x_conv2 = tf.nn.relu(conv2d(x_pool1, W_conv2) + b_conv2)
x_pool2 = max_pooling_2x2(x_conv2)

# Flatten - keras 'flatten'
x_flat = tf.reshape(x_pool2, [-1, 42*48*64])

# Dense fully connected layer
W_fc1 = weight_variable([42 * 48 * 64, 1024])
b_fc1 = bias_variable([1024])
x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

# Regularization with dropout
keep_prob = tf.placeholder(tf.float32)
x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

# Classification layer
W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])
y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2



y = tf.nn.softmax(y_conv)


# Loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# Setup to test accuracy of model
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.global_variables_initializer())

# Train model
# Run once to get the model to a good confidence level
for i in range(tt/100+1):
    end = min((i+1)*100,tt)
    batch_x = x_images[i*100:end]
    batch_y = y_labels[i*100:end]
    #if i%2 == 0:

    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.4})
    train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

print("test accuracy %g"%accuracy.eval(feed_dict={x: x_images[50:150], 
                                                  y_: y_labels[50:150], keep_prob: 1.0}))



#------------------------------------
def plot_predictions(image_list, output_probs=False, adversarial=False):
    '''
    Evaluate images against trained model and plot images.
    If adversarial == True, replace middle image title appropriately
    Return probability list if output_probs == True
    '''
    prob = y.eval(feed_dict={x: image_list, keep_prob: 1.0})
    
    pred_list = np.zeros(len(image_list)).astype(int)
    pct_list = np.zeros(len(image_list)).astype(int)
    
    #Setup image grid
    import math
    cols = 3
    rows = int(math.ceil(image_list.shape[0]/cols))
    fig = plt.figure(1, (12., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )
    
    # Get probs, images and populate grid
    for i in range(len(prob)):
        pred_list[i] = np.argmax(prob[i]) # here index == classification
        pct_list[i] = prob[i][pred_list[i]] * 100

        image = image_list[i].reshape(168,192)
        grid[i].imshow(image)
        
        grid[i].set_title('Label: {0} \nCertainty: {1}%' \
                          .format(pred_list[i], 
                                  pct_list[i]))
        
        # Only use when plotting original, partial deriv and adversarial images
        if (adversarial) & (i % 3 == 1): 
            grid[i].set_title("Adversarial \nPartial Derivatives")
        
    plt.show()
    
    return prob if output_probs else None
#----------------------------------------



def create_plot_adversarial_images(x_image, y_label, lr=0.1, n_steps=1, output_probs=False):
    
    original_image = x_image
    probs_per_step = []
    
    # Calculate loss, derivative and create adversarial image
    loss =  tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_conv)
    deriv = tf.gradients(loss, x)
    image_adv = tf.stop_gradient(x - tf.sign(deriv)*lr/n_steps)
    image_adv = tf.clip_by_value(image_adv, 0, 1) # prevents -ve values creating 'real' image
    
    for _ in range(n_steps):
        # Calculate derivative and adversarial image
        dydx = sess.run(deriv, {x: x_image, keep_prob: 1.0}) # can't seem to access 'deriv' w/o running this
        x_adv = sess.run(image_adv, {x: x_image, keep_prob: 1.0})
        
        # Create darray of 3 images - orig, noise/delta, adversarial
        x_image = np.reshape(x_adv, (1, 32256))
        img_adv_list = original_image
        img_adv_list = np.append(img_adv_list, dydx[0], axis=0)
        img_adv_list = np.append(img_adv_list, x_image, axis=0)

        # Print/plot images and return probabilities
        probs = plot_predictions(img_adv_list, output_probs=output_probs, adversarial=True)
        #print np.argmax(probs, axis=1)
        probs_per_step.append(probs) if output_probs else None
    
    return probs_per_step
   



# Pick a random second person's image from first 1000 images 
# Create adversarial image and with target label 0
index_of_2s = np.nonzero(y_labels[0:1000][:,1])[0]
rand_index = np.random.randint(0, len(index_of_2s))
image_norm = x_images[index_of_2s[rand_index]]
image_norm = np.reshape(image_norm, (1, 32256))
label_adv = [0,0,0,1,0]


# Plot adversarial images
# Over each step, model certainty changes from person 2 to person 1
create_plot_adversarial_images(image_norm, label_adv, lr=0.2, n_steps=5)

 
sess.close()
