import os, sys
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
from run_nerf_helpers import *

import warnings
warnings.filterwarnings('ignore')


def posenc(x):
  rets = [x]
  for i in range(L_embed):
    for fn in [tf.sin, tf.cos]:
      rets.append(fn(2.**i * x))
  return tf.concat(rets, -1)


L_embed = 6          # 0
embed_fn = posenc    # tf.identity


def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    # Compute 3D query points
    z_vals = tf.linspace(near, far, N_samples)
    if rand:
      z_vals += tf.random.uniform(list(rays_o.shape[:-1]) + [N_samples]) * (far-near)/N_samples
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    # Run network
    pts_flat = tf.reshape(pts, [-1,3])
    pts_flat = embed_fn(pts_flat)
    raw = batchify(network_fn, chunk=1024*32)(pts_flat)
    raw = tf.reshape(raw, list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    sigma_a = tf.nn.relu(raw[...,3])
    rgb = tf.math.sigmoid(raw[...,:3])

    # Do volume rendering
    dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], tf.broadcast_to([1e10], z_vals[...,:1].shape)], -1)
    alpha = 1.-tf.exp(-sigma_a * dists)
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    rgb_map = tf.reduce_sum(weights[...,None] * rgb, -2)
    depth_map = tf.reduce_sum(weights * z_vals, -1)
    acc_map = tf.reduce_sum(weights, -1)

    return rgb_map, depth_map, acc_map


def run_tiny_nerf():
    if not os.path.exists('data/nerf_data.npz'):
        print("ERROR file not found")
        return

    print("Loading images/poses")

    data = np.load('data/nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = images.shape[1:3]
    num = images.shape[0]
    print(images.shape, poses.shape, focal)

    # make test image from image of the middle
    test_idx = int(0.5 * num);
    testimg, testpose = images[test_idx], poses[test_idx]

    # remove test image from train set
    images[test_idx:-1, :] = images[test_idx+1:, :]
    poses[test_idx:-1, :] = poses[test_idx+1:, :]
    images = images[:-1]
    poses = poses[:-1]
    print(images.shape, poses.shape, focal)


    print("Start trainning...")
    model = init_tiny_nerf_model(L_embed)
    optimizer = tf.keras.optimizers.Adam(5e-4)

    N_samples = 32
    N_iters = 100000
    psnrs = []
    iternums = []
    i_plot = 200

    basedir = "data/logs/"
    expname = "exp"
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    def save_weights(net, prefix, i):
        path = os.path.join(
            basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
        np.save(path, net.get_weights())
        print('saved weights at', path)

    import time
    t = time.time()
    for i in range(N_iters+1):
        img_i = np.random.randint(images.shape[0])
        target = images[img_i]
        pose = poses[img_i]
        rays_o, rays_d = get_rays(H, W, focal, pose)
        with tf.GradientTape() as tape:
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, rand=True)
            loss = tf.reduce_mean(tf.square(rgb - target))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i%i_plot==0:
            save_weights(model, "", i);

            print(i, (time.time() - t) / i_plot, 'secs per iter')
            t = time.time()

            # Render the holdout view for logging
            rays_o, rays_d = get_rays(H, W, focal, testpose)
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
            loss = tf.reduce_mean(tf.square(rgb - testimg))
            psnr = -10. * tf.math.log(loss) / tf.math.log(10.)

            psnrs.append(psnr.numpy())
            iternums.append(i)

            plt.figure(figsize=(20,4))
            plt.subplot(131)
            plt.imshow(rgb)
            plt.title(f'Iteration: {i}')
            plt.subplot(132)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            plt.subplot(133)
            plt.imshow(testimg)
            plt.title('gt image')
            plt.savefig(basedir + str(i) + ".png")


if __name__ == '__main__':
    run_tiny_nerf()
