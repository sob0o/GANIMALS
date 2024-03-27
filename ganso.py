from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Activation, LeakyReLU, BatchNormalization, Input, Dropout, Flatten, Dense, Reshape
from keras.models import Model
from keras.initializers import RandomNormal
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os 
from keras.utils import plot_model



class GAN():
    def __init__(self):
        self.weights_init = RandomNormal(mean=0, stddev=0.02)
        self.z_dim = 100
        self.generator_dense_layer = (7, 7, 64)
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
        self.learning_rate = 0.0008
        self.batch_size = 64
        self.epoch = 0

        self._build_discriminator()
        self._build_generator()
        self._build_adversarial()

    def _build_discriminator(self):
        discriminator_input = Input((28, 28, 1))
        x = discriminator_input

        x = Conv2D(
            kernel_size=(5, 5),
            strides=2,
            padding='same',
            filters=64,
            name="conv1"
        )(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.4)(x)

        x = Conv2D(
            kernel_size=(5, 5),
            padding='same',
            strides=2,
            name='conv2',
            filters=64,
            kernel_initializer=self.weights_init
        )(x)
        x = LeakyReLU(alpha=0.30)(x)
        x = Dropout(rate=0.4)(x)

        x = Conv2D(
            kernel_size=5,
            strides=2,
            padding='same',
            name='conv3',
            filters=128,
            kernel_initializer=self.weights_init
        )(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(rate=0.4)(x)

        x = Conv2D(
            filters=128,
            kernel_size=5,
            strides=1,
            padding='same',
            name='conv4',
            kernel_initializer=self.weights_init
        )(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(rate=0.4)(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        self.discriminator_model = Model(discriminator_input, x)

    def _build_generator(self):
        generator_input = Input(shape=(self.z_dim,))
        x = generator_input

        x = Dense(np.prod(self.generator_dense_layer))(x)

        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = Reshape((7, 7, 64))(x)

        x = UpSampling2D()(x)
        x = Conv2D(
            filters=128,
            kernel_size=5,
            strides=1,
            padding='same'
        )(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2D(
            filters=64,
            kernel_size=5,
            strides=1,
            padding='same'
        )(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(
            filters=64,
            kernel_size=5,
            strides=1,
            padding='same'
        )(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(
            filters=1,
            kernel_size=5,
            padding='same',
            strides=1
        )(x)

        output_generator = Activation('tanh')(x)
        self.generator_model = Model(generator_input, output_generator)

    def get_opti(self, lr):
        return Adam(lr=lr, beta_1=0.5)

    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def _build_adversarial(self):
        self.discriminator_model.compile(
            optimizer=Adam(lr=self.learning_rate, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.set_trainable(self.discriminator_model, False)

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator_model(self.generator_model(model_input))
        self.model = Model(model_input, model_output)

        self.model.compile(
            optimizer=Adam(lr=self.learning_rate, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.set_trainable(self.discriminator_model, True)

    def train_discriminator(self, x_train, using_generator):
        batch_size = self.batch_size
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator_model.predict(noise)

        d_loss_real, d_acc_real = self.discriminator_model.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake = self.discriminator_model.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)
        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train_generator(self):
        batch_size = self.batch_size
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)



    def train(self, x_train, epochs, print_every_n_batches=1, using_generator=False):
        batch_size = self.batch_size
        plot = True
        for self.epoch in range(0, epochs):
            d = self.train_discriminator(x_train=x_train, using_generator=False)
            g = self.train_generator()
            print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (self.epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))
            self.d_losses.append(d)
            self.g_losses.append(g)
            
            if  plot :
                self.plot_model('run/logi')
                plot_model(self.generator_model,to_file=os.path.join('run/logi' ,'viz/generator.png'), show_shapes = True, show_layer_names = True)
                

            if self.epoch % print_every_n_batches == 0:
                self.sample_images(run_folder='run/logi')
                self.save_model(run_folder='run/logi')


    def sample_images(self, run_folder):
        
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator_model.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15,15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray')
                axs[i,j].axis('off')
                cnt += 1
        
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        print("IMAGES-saved " + str(self.epoch) )
        plt.close()

    def save_model (self,run_folder):
        self.model.save(os.path.join(run_folder,'model.h5'))
        self.generator_model.save(os.path.join(run_folder,'generator.h5'))

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)