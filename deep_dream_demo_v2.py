
# boilerplate code
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
#from IPython.display import clear_output, Image, display, HTML


import tensorflow as tf

#!wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip inception5h.zip

model_fn = 'tensorflow_inception_graph.pb'

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))


#***************************************************

# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139  # picking some feature channel to visualize

# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0


def showarray(a, fname, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    #f = BytesIO()

    PIL.Image.fromarray(a).save(fname, fmt)
    #display(Image(data=f.getvalue()))



def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5


def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0" % layer)


def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj)  # defining the optimization objective. This is mean of a given channel in a tensor layer defined by t_obj
    # we want to maaximize this objective

    t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

    img = img0.copy()
    showarray(visstd(img),'./results/result_0')

    act_obj = sess.run(t_obj, {t_input: img_noise})
    print('objective tensor size', act_obj.shape)

    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input: img})
        # normalizing the gradient, so the same step size should work
        g /= g.std() + 1e-8  # for different layers and networks
        img += g * step
        print(i, ' ', score)

        fname='./results/result_'+str(i)
        showarray(visstd(img),fname)
        # clear_output()
    showarray(visstd(img),'./results/result_final')

render_naive(T(layer)[:, :, :, channel])

#***************************************************


#***************************************************

#Multiscale image generation
# We are going to apply gradient ascent on multiple scales.
# Details formed on smaller scale will be upscaled and augmented with additional details on the next scale.

#With multiscale image generation it may be tempting to set the number of octaves to some high value to produce wallpaper-sized images.
# Storing network activations and backprop values will quickly run out of GPU memory in this case. There is a simple trick to avoid this:
# split the image into smaller tiles and compute each tile gradient independently.
#Applying random shifts to the image before every iteration helps avoid tile seams and improves the overall image quality.

print('**** Multiscale ****')

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, t_score, t_obj, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    #print('tile size', tile_size)

    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)

    y=0
    x=0
    sub = img_shift[y:y + sz, x:x + sz]
    act_obj = sess.run(t_obj, {t_input: sub})
    print('objective tensor size', act_obj.shape)

    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g, score = sess.run([t_grad,t_score ], {t_input: sub})
            #score = sess.run(t_score, {input: sub})
            grad[y:y+sz,x:x+sz] = g
            #print('x:',x,'y:',y)


            print('score: ', score)



    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_multiscale(t_obj, img0=img_noise, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

    img = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            hw = np.float32(img.shape[:2]) * octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad, t_score, t_obj)
            # normalizing the gradient, so the same step size should work
            g /= g.std() + 1e-8  # for different layers and networks
            img += g * step
            print('o: ' ,octave,'i: ',i, 'size:', g.shape, end=' ')
            # clear_output()


            fname = './results/multi_scale_result_' + str(i)+ '_'+str(octave)
            showarray(visstd(img), fname)


render_multiscale(T(layer)[:, :, :, channel])

#***************************************************
#Laplacian Pyramid Gradient Normalization
#This looks better, but the resulting images mostly contain high frequencies. Can we improve it?
# One way is to add a
# smoothness prior into the optimization objective. This will effectively blur the image a little every iteration, suppressing
# the higher frequencies, so that the lower frequencies can catch up.
#This will require more iterations to produce a nice image. Why don't we just boost lower frequencies of the gradient instead?
#                                                                                                                                                                                                                                                                                                                                                                                        ' One way to achieve this is through the Laplacian pyramid decomposition. We call the resulting technique Laplacian Pyramid Gradient Normalization.
#*******************************************

print('**** Laplace ***')

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]

def render_lapnorm(t_obj, img0=img_noise, visfunc=visstd,
                   iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad, t_score, t_obj)
            g = lap_norm_func(g)
            img += g*step
            #print('.', end = ' ')
            print('o: ', octave, 'i: ', i, 'size:', g.shape, end=' ')

            fname = './results/laplace_result_' + str(i) + '_' + str(octave)
            showarray(visstd(img), fname)


render_lapnorm(T(layer)[:,:,:,channel])




#************************************************************
# deap dream
#**************************************************************

print('**** deep dream ****')

def render_deepdream(t_obj, img0=img_noise,
                     iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad,t_score, t_obj)
            img += g * (step / (np.abs(g).mean() + 1e-7))
            print('.', end=' ')
            # clear_output()

            fname = './results/deep_dream_result_' + str(i) + '_' + str(octave)
            showarray(img / 255.0, fname)



#img0 = PIL.Image.open('pilatus800.jpg')
img0 = PIL.Image.open('mmd_22000_01_9000.png')

img0 = np.float32(img0)
#showarray(img0/255.0)
#render_deepdream(tf.square(T('mixed4c')), img0)

render_deepdream(T(layer)[:,:,:,139], img0, iter_n=20, step=5, octave_n=4)