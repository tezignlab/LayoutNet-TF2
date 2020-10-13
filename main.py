import os
from PIL import Image
import time
import numpy as np
import tensorflow as tf
from model import *
import config
from datetime import datetime
from dataset import Dataset
import argparse

# load dataset
dataset = Dataset()

# create model
layoutnet = LayoutNet(config)

# define loss functions


# Discriminator loss is same as the least square GAN (LSGAN)
def discriminator_loss(D_real, D_fake):
    loss_D_real = tf.reduce_mean(tf.nn.l2_loss(D_real - tf.ones_like(D_real)))
    loss_D_fake = tf.reduce_mean(tf.nn.l2_loss(D_fake - tf.zeros_like(D_fake)))

    loss_D = loss_D_real + loss_D_fake

    return loss_D


# Generator loss should have 3 parts:
# 1. loss_Gls - G loss in LSGAN
# 2. recon_loss - reconstruction loss
# 3. Variety Loss mentioned in the paper, but not implemented in author's code
def generator_loss(x, z_log_sigma_sq, z_mean, D_fake, G_recon):
    # loss_Gls is the loss function for G in LSGAN
    loss_Gls = tf.reduce_mean(tf.nn.l2_loss(D_fake - tf.ones_like(D_fake)))

    orig_input = tf.reshape(x, [config.batch_size, 64 * 64 * 3])
    orig_input = (orig_input + 1) / 2.
    generated_flat = tf.reshape(G_recon, [config.batch_size, 64 * 64 * 3])
    generated_flat = (generated_flat + 1) / 2.

    recon_loss = tf.reduce_sum(tf.pow(generated_flat - orig_input, 2), 1)

    recon_loss = tf.reduce_mean(recon_loss) / 64 / 64 / 3

    loss_G = loss_Gls + recon_loss

    return loss_G


# Encoder loss should have 2 parts:
# 1. kl_div - KL divergence loss
# 2. recon_loss - reconstruction loss
def encoder_loss(x, z_log_sigma_sq, z_mean, G_recon):
    kl_div = -0.5 * tf.reduce_sum(
        1 + 2 * z_log_sigma_sq - tf.square(z_mean) -
        tf.exp(2 * z_log_sigma_sq), 1)

    orig_input = tf.reshape(x, [config.batch_size, 64 * 64 * 3])
    orig_input = (orig_input + 1) / 2.
    generated_flat = tf.reshape(G_recon, [config.batch_size, 64 * 64 * 3])
    generated_flat = (generated_flat + 1) / 2.

    recon_loss = tf.reduce_sum(tf.pow(generated_flat - orig_input, 2), 1)

    loss_E = tf.reduce_mean(kl_div + recon_loss) / 64 / 64 / 3

    return loss_E


# define optimizer
# all optimizer use Adam

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr,
                                               beta_1=config.beta1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr,
                                                   beta_1=config.beta1)
encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr,
                                             beta_1=config.beta1)

# define checkpoint and manager

checkpoint_prefix = os.path.join(config.checkpoint_dir,
                                 config.checkpoint_basename)
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    encoder_optimizer=encoder_optimizer,
    generator=layoutnet.generator,
    discriminator=layoutnet.discriminator,
    encoder=layoutnet.encoder,
    embeddingSemvec=layoutnet.embeddingSemvec,
    embeddingImg=layoutnet.embeddingImg,
    embeddingTxt=layoutnet.embeddingTxt,
    embeddingFusion=layoutnet.embeddingFusion)
    
manager = tf.train.CheckpointManager(checkpoint,
                                     config.checkpoint_dir,
                                     max_to_keep=None)

# define the training loop


def train_step(z,
               is_training=True,
               discriminator=True,
               generator=True,
               encoder=True):
    # get data from dataset
    resized_image, label, textRatio, imgRatio, visualfea, textualfea = \
        dataset.next()

    with tf.GradientTape(persistent=True) as tape:
        # get all results from the model
        z_mean, z_log_sigma_sq, E, G, G_recon, D_real, D_fake = layoutnet(
            resized_image,
            label,
            textRatio,
            imgRatio,
            visualfea,
            textualfea,
            z,
            is_training=is_training)

        # calculate loss
        disc_loss = discriminator_loss(D_real, D_fake)
        gen_loss = generator_loss(x=resized_image,
                                  z_log_sigma_sq=z_log_sigma_sq,
                                  z_mean=z_mean,
                                  D_fake=D_fake,
                                  G_recon=G_recon)
        encod_loss = encoder_loss(x=resized_image,
                                  z_log_sigma_sq=z_log_sigma_sq,
                                  z_mean=z_mean,
                                  G_recon=G_recon)

    if is_training:
        # get the trainable variables
        # BP algorithm will modify these variables
        discriminator_variables = layoutnet.discriminator.trainable_variables+ \
                            layoutnet.embeddingImg.trainable_variables + \
                            layoutnet.embeddingTxt.trainable_variables + \
                            layoutnet.embeddingSemvec.trainable_variables + \
                            layoutnet.embeddingFusion.trainable_variables
        generator_variables = layoutnet.generator.trainable_variables
        # when training encoder, we not only modify encoder's weights
        # but also modify weights of embedding layers
        encoder_variables = layoutnet.encoder.trainable_variables + \
                            layoutnet.embeddingImg.trainable_variables + \
                            layoutnet.embeddingTxt.trainable_variables + \
                            layoutnet.embeddingSemvec.trainable_variables + \
                            layoutnet.embeddingFusion.trainable_variables

        # there will be 3 kinds of training
        # we use flags to control training process

        # 1. train disciminator
        if discriminator:
            gradients_of_discriminator = tape.gradient(
                disc_loss, discriminator_variables)
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, discriminator_variables))

        # 2. train generator
        if generator:
            gradients_of_generator = tape.gradient(gen_loss,
                                                       generator_variables)
            generator_optimizer.apply_gradients(
                zip(gradients_of_generator, generator_variables))

        # 3. train encdoer
        if encoder:
            gradients_of_encoder = tape.gradient(
                encod_loss, encoder_variables)
            encoder_optimizer.apply_gradients(
                zip(gradients_of_encoder, encoder_variables))

    return disc_loss, gen_loss, encod_loss


# sampling function
def sample(step, sample_dir='./sample'):
    layout_path = os.path.join(sample_dir, 'layout')
    img_fea_path = os.path.join(sample_dir, 'visfea')
    txt_fea_path = os.path.join(sample_dir, 'texfea')
    sem_vec_path = os.path.join(sample_dir, 'semvec')

    # read images' name list from local file
    f = open(os.path.join(sample_dir, 'imgSel_128.txt'), 'r')
    name = f.read()
    name_list = name.split()

    n_samples = len(name_list)

    for i in range(n_samples):
        name_tmp = name_list[i]

        # get layout annotation (image) according to name
        layout_name = os.path.join(layout_path, name_tmp)
        img = Image.open(layout_name)
        rgb = np.array(img).reshape(1, 64, 64, 3)
        rgb = rgb.astype(np.float32) * (1. / 127.5) - 1.  # normalize
        test_layout = np.concatenate(
            (test_layout, rgb), axis=0) if i > 0 else rgb

        # general postfix of features' filename
        file_name = name_tmp[0:-4] + '.npy'

        # get 3 kinds of features from local file

        # image feature
        img_fea_name = os.path.join(img_fea_path, file_name)
        img_fea = np.load(img_fea_name)
        img_fea = img_fea.reshape(1, 14, 14, 512)
        test_img_fea = np.concatenate(
            (test_img_fea, img_fea), axis=0) if i > 0 else img_fea

        # text feature
        txt_fea_name = os.path.join(txt_fea_path, file_name)
        txt_fea = np.load(txt_fea_name)
        txt_fea = txt_fea.reshape((1, 300))
        test_txt_fea = np.concatenate(
            (test_txt_fea, txt_fea), axis=0) if i > 0 else txt_fea

        # attribute feature
        sem_vec_name = os.path.join(sem_vec_path, file_name)
        convar = np.load(sem_vec_name)

        # use np.eye(n) and feature value to generate one-hot encoding
        category_input = np.eye(6)[int(convar[0, 0])].reshape([1, 6])
        text_ratio_input = np.eye(7)[int(convar[0, 1])].reshape([1, 7])
        img_ratio_input = np.eye(10)[int(convar[0, 2])].reshape([1, 10])

        sem_vec = np.concatenate(
            [category_input, text_ratio_input, img_ratio_input], 1)
        sem_vec = sem_vec.astype(np.float32)
        test_sem_vec = np.concatenate(
            (test_sem_vec, sem_vec), axis=0) if i > 0 else sem_vec

    # start sampling

    # get fixed random value z
    random_z_val = np.load(os.path.join(sample_dir, 'noiseVector_128.npy'))

    # start calculating
    # the process is same as training
    test_sem_vec = layoutnet.embeddingSemvec(test_sem_vec, is_training=False)
    test_img_fea = layoutnet.embeddingImg(test_img_fea, is_training=False)
    test_txt_fea = layoutnet.embeddingTxt(test_txt_fea, is_training=False)
    test_label = layoutnet.embeddingFusion(test_sem_vec,
                                           test_img_fea,
                                           test_txt_fea,
                                           is_training=False)
    test_dis_label = tf.reshape(test_label,
                                shape=[-1, 1, 1, config.latent_dim]) * tf.ones(
                                    [128, 4, 4, config.latent_dim])
    test_mean, test_log_sigma_sq = layoutnet.encoder(test_layout,
                                                     is_training=False,
                                                     y=test_dis_label)
    E_input = test_mean + tf.exp(test_log_sigma_sq) * random_z_val
    G = layoutnet.generator(E_input, is_training=False, y=test_label)
    G = (G + 1.) / 2.

    # save result as image
    im_name = os.path.join(config.sampledir, 'sample_%d.jpg' % (step + 1))
    h, w = G.shape[1], G.shape[2]
    merge_img = np.zeros((h * 16, w * 8, 3))
    for idx, image in enumerate(G):
        i = idx % 8
        j = idx // 8
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
    img = Image.fromarray(np.uint8(merge_img * 255))
    img.save(im_name)


def train():
    for step in range(config.max_steps):
        t1 = time.time()

        z = np.random.normal(0.0, 1.0, size=(config.batch_size,
                                             config.z_dim)).astype(np.float32)

        # train disc
        disc_loss, gen_loss, encod_loss = train_step(z=z,
                                                     discriminator=True,
                                                     generator=False,
                                                     encoder=False)

        # train gen
        disc_loss, gen_loss, encod_loss = train_step(z=z,
                                                     discriminator=False,
                                                     generator=True,
                                                     encoder=False)

        # train encoder
        disc_loss, gen_loss, encod_loss = train_step(z=z,
                                                     discriminator=False,
                                                     generator=False,
                                                     encoder=True)

        # train gen and encoder
        # make sure discriminator cannot distinguish fake images
        while disc_loss < 1.:
            disc_loss, gen_loss, encod_loss = train_step(z,
                                                         discriminator=False,
                                                         generator=True,
                                                         encoder=True)

        t2 = time.time()

        if (step + 1) % config.summary_every_n_steps == 0:
            disc_loss, gen_loss, encod_loss = train_step(z, is_training=False)
            print("step {:5d},loss = (G: {:.8f}, D: {:.8f}), E: {:.8f}".format(
                step + 1, gen_loss, disc_loss, encod_loss))

        if (step + 1) % config.sample_every_n_steps == 0:
            eta = (t2 - t1) * (config.max_steps - step + 1)
            print("Finished {}/{} step, ETA:{:.2f}s".format(
                step + 1, config.max_steps, eta))

            manager.save()

            # get and save samples
            sample(step)


def test():
    # restore latest checkpoint
    # checkpoint.restore(manager.latest_checkpoint)

    layoutnet.load_weights('./checkpoints/ckpt-300')

    # run sample function to generatre sample using checkpoint

    # layoutnet.save_weights('./model/result')
    sample(step=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    result = parser.parse_args()

    if result.train and result.test:
        raise ValueError(
            'train flag and test flag cannot be set at the same time.')
    elif result.train:
        train()
    elif result.test:
        test()
    else:
        raise ValueError('you must add flag "--test" or "--train" to run.')
