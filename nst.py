
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import importlib
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras import metrics
from keras.models import Model
from keras.applications import VGG16
from PIL import Image
import keras.backend as K


# In[2]:

resultspath = '<directory where you want to store the images>'


# In[3]:

img = Image.open('hugo.jpg')
img.size


# In[4]:

mean_sub = np.array([123.68, 116.779, 103.939], dtype=np.float32)
pre_processing = lambda x: (x - mean_sub)[:,:,:,::-1]


# In[5]:

de_preprocess = lambda x, shape: np.clip(x.reshape(shape)[:,:,:,::-1] + mean_sub, 0, 255)


# In[6]:

img_arr = pre_processing(np.expand_dims(np.array(img), 0))
shape_content = img_arr.shape

shape_content


# In[23]:

model = VGG16(weights='imagenet', include_top=False)


# In[24]:

layer = model.get_layer('block2_conv2').output


# In[25]:

layer


# In[26]:

layer_model = Model(model.input, layer)


# In[87]:

target = K.variable(layer_model.predict(img_arr))
content_target = target

content_target


# In[7]:

class Evaluator(object):
    def __init__(self, f, shp): self.f, self.shp = f, shp
        
    def loss(self, x):
        loss_, self.grad_values = self.f([x.reshape(self.shp)])
        return loss_.astype(np.float64)

    def grads(self, x): return self.grad_values.flatten().astype(np.float64)


# In[29]:

loss = metrics.mse(layer, target)
grads = K.gradients(loss, model.input)
fn = K.function([model.input], [loss] + grads)
evaluator = Evaluator(fn, shape_content)


# In[8]:

def solve_image(eval_obj, niter, x, path):
    for i in range(niter):
        x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                         fprime=eval_obj.grads, maxfun=20)
        x = np.clip(x, -127, 127)
        print ('Minimum Loss Value:', min_val)
        imsave('{}res_at_iteration_{}.png'.format(path, i), de_preprocess(x.copy(), shape_content)[0])
    return x


# In[31]:

def rand_img(shape):
    return np.random.uniform(-2.5, 2.5, shape) / 100

x = rand_img(shape_content)
plt.imshow(x[0])

x.shape, img_arr.shape


# In[32]:

x = solve_image(evaluator, 10, x, resultspath+'content/')


# In[26]:

style = Image.open('starry_night.jpg')
if style.size != img.size:
    style = style.resize(img.size, Image.ANTIALIAS)
style


# In[27]:

style_arr = pre_processing(np.expand_dims(np.array(style), 0))
style_shape = style_arr.shape
style_shape


# In[28]:

model = VGG16(weights='imagenet', include_top=False, input_shape=style_shape[1:])


# In[29]:

outputs = {layer.name:layer.output for layer in model.layers}
outputs


# In[30]:

temp = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
style_layers = [outputs[i] for i in temp]
style_layers


# In[31]:

style_model = Model(model.input, style_layers)


# In[32]:

style_target = [K.variable(i) for i in style_model.predict(style_arr)]
style_target


# In[33]:

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()


# In[34]:

def style_loss(x, targ):
    return metrics.mse(gram_matrix(x), gram_matrix(targ))


# In[78]:

loss = sum(style_loss(l[0], t[0]) for l, t in zip(style_layers, style_target))
grads = K.gradients(loss, model.input)
fn = K.function([model.input], [loss] + grads)
evaluator = Evaluator(fn, style_shape)


# In[18]:

import scipy

def rand_img(shape):
    return np.random.uniform(-2.5, 2.5, shape)

x = rand_img(style_shape)
# Guassian Blur
# Helps in reducing image noise and image detail
# x = scipy.ndimage.filters.gaussian_filter(x, [0,2,2,0])
plt.imshow(x[0])

x.shape, style_shape


# In[85]:

stylize(evaluator, 10, x, resultspath+'style/')


# In[86]:

Image.open(resultspath+'style/res_at_iteration_9.png')


# ## Style Transfer

# In[35]:

shape_content, style_shape


# In[36]:

content_layer = outputs['block2_conv2']
content_model = Model(model.input, content_layer)
content_target = content_model.predict(img_arr)


# In[37]:

style_weights = [0.05,0.2,0.2,0.25,0.3]
total_loss = sum(style_loss(l[0], t[0])*w for l, t, w in zip(style_layers, style_target, style_weights))
total_loss += metrics.mse(content_layer, content_target)/10

grads = K.gradients(total_loss, model.input)
fn = K.function([model.input], [total_loss] + grads)
evaluator = Evaluator(fn, style_shape)


# In[38]:

x = rand_img(style_shape)
plt.imshow(x[0])


# In[39]:

stylize(evaluator, 10, x, resultspath+'starry_baby_transfer/')


# In[ ]:



