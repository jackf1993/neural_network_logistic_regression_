#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install deepchem


# In[6]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate synthetic data
N = 100
# Zeros form a Gaussian centered at (-1, -1)
x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(N//2,))
y_zeros = np.zeros((N//2,))



# Ones form a Gaussian centered at (1, 1)
x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)), cov=.1*np.eye(2), size=(N//2,))
y_ones = np.ones((N//2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])

# Cast input data to float32
x_np = x_np.astype(np.float32)
y_np = y_np.astype(np.float32)

# Ensure model parameters are float32 tensors
W = tf.Variable(tf.random.normal((2, 1), dtype=tf.float32))
b = tf.Variable(tf.random.normal((1,), dtype=tf.float32))

# Plot the data
plt.scatter(x_zeros[:,0], x_zeros[:,1], c='r', marker='x', label='y=0')
plt.scatter(x_ones[:,0], x_ones[:,1], c='b', marker='o', label='y=1')
plt.legend()
plt.title("Synthetic Data")
plt.show()

# Define model
class LogisticRegression(tf.keras.Model):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.W = tf.Variable(tf.random.normal((2, 1)))
        self.b = tf.Variable(tf.random.normal((1,)))

    def call(self, inputs):
        return tf.squeeze(tf.matmul(inputs, self.W) + self.b)

# Create model instance
model = LogisticRegression()

# Loss function
def loss_fn(model, inputs, targets):
    y_logit = model(inputs)
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=targets))

# Optimization
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training loop
for i in range(20000):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x_np, y_np)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if i % 1000 == 0:
        print("step %d, loss: %f" % (i, loss))

# Plot the predicted outputs on top of the data
y_pred_np = tf.round(tf.sigmoid(model(x_np))).numpy().flatten()
plt.scatter(x_np[:, 0], x_np[:, 1], c=y_pred_np, cmap="coolwarm")
plt.show()


# In[ ]:




