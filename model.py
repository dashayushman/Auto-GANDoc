import tensorflow as tf
import tensorflow.contrib.slim as slim
from Utils import ops

class AutoGAN:
	'''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
	'''

	def __init__(self, options) :
		self.options = options

	def build_model(self) :

		print('Initializing placeholder')
		img_size = self.options['image_size']
		image = tf.placeholder('float32', [self.options['batch_size'],
		                              img_size, img_size, 3],
		                              name = 'input_image')

		training = tf.placeholder(tf.bool, name='training')

		print('Building the Encoder')
		en_image, en_image_shape = self.encoder(image,
							self.options['n_classes'], training)

		print('Building the Generator')
		gen_image = self.decoder(en_image, training)

		flat_image = tf.contrib.layers.flatten(image)
		flat_gen_image = tf.contrib.layers.flatten(gen_image)

		print('Building the Loss Function')
		auto_gan_loss = tf.contrib.losses.mean_squared_error(flat_gen_image,
															labels=flat_image)
		t_vars = tf.trainable_variables()
		self.add_tb_histogram_summaries(t_vars)
		self.add_tb_scalar_summaries(auto_gan_loss)

		self.add_image_summary('Encoded Images', image,
		                       self.options['batch_size'])
		self.add_image_summary('Decoded Images', gen_image,
							   self.options['batch_size'])


		ag_vars = [var for var in t_vars if 'e_' in var.name or 'd_' in
				   var.name]

		input_tensors = {
			'input_images' : image,
			'training' : training,
		}

		variables = {
			'ag_vars' : ag_vars
		}

		loss = {
			'autogan_loss' : auto_gan_loss,
		}

		outputs = {
			'decoder' : gen_image
		}

		checks = {
			'auto_gan_loss': auto_gan_loss,
			'decoder': gen_image
		}

		return input_tensors, variables, loss, outputs, checks

	def add_tb_histogram_summaries(self, t_vars):

		print('List of all variables')
		for v in t_vars:
			print(v.name)
			print(v)
			self.add_histogram_summary(v.name, v)

	def add_tb_scalar_summaries(self, autogan_loss):
		self.add_scalar_summary("AutoGAN_Loss", autogan_loss)

	def add_scalar_summary(self, name, var):
		tf.summary.scalar(name, var)

	def add_histogram_summary(self, name, var):
		with tf.name_scope('summaries'):
			tf.summary.histogram(name, var)

	def add_image_summary(self, name, var, max_outputs=1):
		with tf.name_scope('summaries'):
			tf.summary.image(name, var, max_outputs=max_outputs)

	# GENERATOR IMPLEMENTATION based on :
	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def decoder(self, en_image, t_training):

		s = self.options['image_size']
		s2, s4, s8, s16, s32 = int(s / 2), int(s / 4), int(s / 8),\
							   int(s / 16), int(s / 32)

		h1 = ops.deconv2d(en_image, [self.options['batch_size'], s16, s16,
		                       self.options['df_dim'] * 4], name = 'd_h1')
		h1 = tf.nn.relu(slim.batch_norm(h1, is_training = t_training,
		                                scope="d_bn1"))

		h2 = ops.deconv2d(h1, [self.options['batch_size'], s8, s8,
		                       self.options['df_dim'] * 6], name = 'd_h2')
		h2 = tf.nn.relu(slim.batch_norm(h2, is_training = t_training,
		                                scope="d_bn2"))
		
		h3 = ops.deconv2d(h2, [self.options['batch_size'], s4, s4,
		                       self.options['df_dim'] * 6], name = 'd_h3')
		h3 = tf.nn.relu(slim.batch_norm(h3, is_training = t_training,
		                                scope="d_bn3"))

		h4 = ops.deconv2d(h3, [self.options['batch_size'], s2, s2,
		                       self.options['df_dim'] * 8], name = 'd_h4')
		h4 = tf.nn.relu(slim.batch_norm(h4, is_training = t_training,
		                                scope="d_bn4"))

		h5 = ops.deconv2d(h4, [self.options['batch_size'], s, s, 3],
		                  name = 'd_h5')

		return (tf.tanh(h5) / 2. + 0.5)


	# DISCRIMINATOR IMPLEMENTATION based on :
	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def encoder(self, image, n_classes, t_training, reuse = False) :

		if reuse :
			tf.get_variable_scope().reuse_variables()

		h0 = ops.lrelu(ops.conv2d(image, self.options['ef_dim'] * 8,
								  name = 'e_h0_conv'))  # 64

		h1 = ops.lrelu(slim.batch_norm(ops.conv2d(h0,
		                                     self.options['ef_dim'] * 8,
		                                     name = 'e_h1_conv'),
		                               reuse=reuse,
		                               is_training = t_training,
		                               scope = 'e_bn1'))  # 32

		h2 = ops.lrelu(slim.batch_norm(ops.conv2d(h1,
		                                     self.options['ef_dim'] * 6,
		                                     name = 'e_h2_conv'),
		                               reuse=reuse,
		                               is_training = t_training,
		                               scope = 'e_bn2'))  # 16
		h3 = ops.lrelu(slim.batch_norm(ops.conv2d(h2,
		                                     self.options['ef_dim'] * 4,
		                                     name = 'e_h3_conv'),
		                               reuse=reuse,
		                               is_training = t_training,
		                               scope = 'e_bn3'))  # 8

		h4 = ops.lrelu(slim.batch_norm(ops.conv2d(h3,
												self.options['ef_dim'] * 2,
												name = 'e_h4_conv'),
		                                reuse=reuse,
		                                is_training = t_training,
		                                scope = 'e_bn4'))  # 8

		h4_shape = h4.get_shape().as_list()
		#h4_flat = tf.contrib.layers.flatten(h4)

		#h5 = ops.linear(h4_flat, 1024, 'fl_e_01')
		#h6 = ops.linear(h5, n_classes, 'fl_e_02')
		
		return h4, h4_shape

	# This has not been used used yet but can be used
	def attention(self, decoder_output, seq_outputs, output_size, time_steps,
			reuse=False) :
		if reuse:
			tf.get_variable_scope().reuse_variables()
		ui = ops.attention(decoder_output, seq_outputs, output_size,
		                   time_steps, name = "g_a_attention")

		with tf.variable_scope('g_a_attention'):
			ui = tf.transpose(ui, [1, 0, 2])
			ai = tf.nn.softmax(ui,  dim=1)
			seq_outputs = tf.transpose(seq_outputs, [1, 0, 2])
			d_dash = tf.reduce_sum(tf.mul(seq_outputs, ai), axis=1)
			return d_dash, ai
