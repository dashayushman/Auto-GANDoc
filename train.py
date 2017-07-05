import model
import argparse
import scipy.misc
import os
import shutil
import progressbar
import pickle

import tensorflow as tf
import numpy as np

from os.path import join
from skimage.transform import resize
from tensorflow.examples.tutorials.mnist import input_data

def main(args):
	model_dir, model_chkpnts_dir, model_samples_dir, model_val_samples_dir,\
			model_summaries_dir, history_path = initialize_directories(args)

	data = load_training_data(data_dir=args.data_dir, data_set=args.data_set)
	model_options = {
		'batch_size': args.batch_size,
		'image_size': args.image_size,
		'df_dim': args.df_dim,
		'ef_dim': args.ef_dim,
		'n_classes': args.n_classes
	}
	# Initialize and build the GAN model
	auto_gan = model.AutoGAN(model_options)
	input_tensors, variables, loss, outputs, checks = auto_gan.build_model()

	ag_optim = tf.train.AdamOptimizer(args.learning_rate,
						 beta1=args.beta1).minimize(loss['autogan_loss'],
						 var_list=variables['ag_vars'])

	global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
	merged = tf.summary.merge_all()
	sess = tf.InteractiveSession()

	summary_writer = tf.summary.FileWriter(model_summaries_dir, sess.graph)

	tf.global_variables_initializer().run()
	saver = tf.train.Saver(max_to_keep=10000)
	if args.resume_model:
		load_checkpoint(model_chkpnts_dir, sess, saver)

	history = load_history(history_path)

	def on_epoch_complete(val_loss, train_loss, epoch):
		history['training_losses'].append(train_loss)
		history['validation_losses'].append(val_loss)
		if val_loss <= history['best_loss']:
			history['best_loss'] = val_loss
			history['best_epoch'] = epoch

		print('\nTraining Loss: {}\nValidation Loss: {}\n'
			  'Best Loss: {}\nBest Epoch: {}\n'.format(train_loss,
			   val_loss, history['best_loss'], history['best_epoch']))
		pickle.dump(history, open(history_path, "wb"))

	train(data, args, global_step_tensor, ag_optim, sess, loss, outputs,
		  input_tensors, merged, summary_writer, saver, model_samples_dir,
		  model_chkpnts_dir, model_val_samples_dir, on_epoch_complete)


def load_history(history_path):
	history = None
	if os.path.exists(history_path):
		history = pickle.load(open(history_path, "rb"))
	else:
		history = {'training_losses': [], 'validation_losses': [],
				   'best_loss': float("inf"), 'best_epoch': 0,
				   'validate_every': args.validate_every}
	return history


def load_checkpoint(checkpoints_dir, sess, saver):
	print('Trying to resume training from a previous checkpoint' +
		  str(tf.train.latest_checkpoint(checkpoints_dir)))
	if tf.train.latest_checkpoint(checkpoints_dir) is not None:
		saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
		print('Successfully loaded model. Resuming training.')
	else:
		print('Could not load checkpoints.  Training a new model')


def train(data, args, global_step_tensor, optimizer, sess, loss, outputs,
		  input_tensors, merged_summaries, summary_writer, saver,
		  model_samples_dir, model_chkpnts_dir, model_val_samples_dir,
		  on_epoch_complete):

	global_step = global_step_tensor.eval()
	gs_assign_op = global_step_tensor.assign(global_step)
	if global_step >= args.epochs:
		print('Already trained for {} epochs. If you want to train for more '
			  'number of epochs then set the epochs flag to number that is '
			  'larger than {}'.format(args.epochs, args.epochs))
		exit()
	data.train._epochs_completed = global_step
	for n_e in range(global_step, args.epochs):
		print('Training Epoch {}\n'.format(n_e))
		num_batches = int(data.train.num_examples / args.batch_size)
		bar = progressbar.ProgressBar(redirect_stdout=True,
									  max_value=num_batches)
		batch_count, training_batch_losses = 0, []
		while n_e == data.train.epochs_completed:
			batch = data.train.next_batch(args.batch_size)
			if args.data_set == 'mnist':
				batch = process_mnist_images(batch)

			# Encoder Update
			_, ag_loss, decoded_images, summary = sess.run([optimizer,
					loss['autogan_loss'], outputs['decoder'], merged_summaries],
				   	feed_dict={
					   input_tensors['input_images'].name: batch[0],
					   input_tensors['training'].name: args.train
				   	})
			training_batch_losses.append(ag_loss)
			summary_writer.add_summary(summary, global_step)

			global_step += 1
			sess.run(gs_assign_op)
			bar.update(batch_count)
			batch_count += 1
			if (batch_count % args.save_every) == 0 and batch_count != 0:
				print("\nAG Loss: {}\n".format(ag_loss))
				print("Saving Images and the Model\n")
				save_for_vis(model_samples_dir, batch[0], decoded_images)
				save_path = saver.save(sess, join(model_chkpnts_dir,
						  "latest_model_{}_temp.ckpt".format(args.data_set)))

		mean_training_loss = np.nanmean(training_batch_losses)

		bar.finish()
		save_epoch_model(saver, sess, model_chkpnts_dir, n_e)

		if n_e % args.validate_every == 0:
			mean_val_loss = validate(data, args, loss, sess, input_tensors,
									 model_val_samples_dir, outputs)
			on_epoch_complete(mean_val_loss, mean_training_loss, n_e)


def save_epoch_model(saver, sess, checkpoints_dir, epoch):
	epoch_dir = join(checkpoints_dir, str(epoch))
	if not os.path.exists(epoch_dir):
		os.makedirs(epoch_dir)
	save_path = saver.save(sess, join(epoch_dir,
		  "model_after_{}_epoch_{}.ckpt".format(args.data_set, epoch)))

def validate(data, args, loss, sess, input_tensors, model_val_samples_dir,
			 outputs):

	print('\nValidating Samples\n')
	bar = progressbar.ProgressBar(redirect_stdout=True,
								  max_value=int(
									  data.validation.num_examples /
									  args.batch_size))
	batch_count = 0
	val_batch_losses = []
	while data.validation.epochs_completed == 0:
		batch = data.validation.next_batch(args.batch_size)
		if args.data_set == 'mnist':
			batch = process_mnist_images(batch)

		ag_loss, decoded_images = sess.run(
				[loss['autogan_loss'], outputs['decoder']],
				feed_dict={
					input_tensors['input_images'].name: batch[0],
					input_tensors['training'].name: args.train
				})
		val_batch_losses.append(ag_loss)
		bar.update(batch_count)
		batch_count += 1
		if (batch_count % 50) == 0 and batch_count != 0:
			save_for_vis(model_val_samples_dir, batch[0],
						 decoded_images)
	bar.finish()
	return np.np.nanmean(val_batch_losses)


def process_mnist_images(batch, output_shape=(128, 128)):
	images, output_images = batch[0], []

	for img in images:
		img_gray = np.reshape(img, newshape=(28, 28))
		rgb_img = np.zeros(shape=(28, 28, 3))
		rgb_img[:, :, 0] = img_gray
		rgb_img[:, :, 1] = img_gray
		rgb_img[:, :, 2] = img_gray
		rgb_img = resize(rgb_img, output_shape)
		output_images.append(rgb_img)
	return [output_images, batch[1]]


def load_training_data(data_dir, data_set) :
	datasets_root_dir = join(data_dir, 'datasets')
	if data_set == 'mnist' :
		return input_data.read_data_sets('MNIST_data', one_hot=True)
	elif data_set == 'flowers':
		raise NotImplementedError()
	elif data_set == 'tobacco':
		raise NotImplementedError()
	else :
		raise NotImplementedError()

def initialize_directories(args):
	model_dir = join(args.data_dir, 'training', args.model_name)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	model_chkpnts_dir = join(model_dir, 'checkpoints')
	if not os.path.exists(model_chkpnts_dir):
		os.makedirs(model_chkpnts_dir)

	model_summaries_dir = join(model_dir, 'summaries')
	if not os.path.exists(model_summaries_dir):
		os.makedirs(model_summaries_dir)

	model_samples_dir = join(model_dir, 'samples')
	if not os.path.exists(model_samples_dir):
		os.makedirs(model_samples_dir)

	model_val_samples_dir = join(model_dir, 'val_samples')
	if not os.path.exists(model_val_samples_dir):
		os.makedirs(model_val_samples_dir)

	history_path = os.path.join(model_dir, 'history.pkl')

	return model_dir, model_chkpnts_dir, model_samples_dir, \
		   model_val_samples_dir, model_summaries_dir, history_path


def save_for_vis(data_dir, real_images, generated_images) :
	shutil.rmtree(data_dir)
	os.makedirs(data_dir)
	for i in range(0, len(real_images)):
		real_images_255 = real_images[i]
		scipy.misc.imsave(join(data_dir, 'real_image_{}.jpg'.format(i)),
						  real_images_255)
		fake_images_255 = (generated_images[i, :, :, :])
		scipy.misc.imsave(join(data_dir, 'fake_image_{}.jpg'.format(i)),
						  fake_images_255)

if __name__ == '__main__' :
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size', type=int, default=64,
						help='Batch Size')

	parser.add_argument('--image_size', type=int, default=128,
						help='Image Size a, a x a')

	parser.add_argument('--df_dim', type=int, default=64,
						help='Number of conv in the first layer gen.')

	parser.add_argument('--ef_dim', type=int, default=64,
						help='Number of conv in the first layer discr.')

	parser.add_argument('--n_classes', type=int, default=102,
						help='Number of classes/class labels')

	parser.add_argument('--data_dir', type=str, default="Data",
						help='Data Directory')

	parser.add_argument('--learning_rate', type=float, default=0.0002,
						help='Learning Rate')

	parser.add_argument('--beta1', type=float, default=0.5,
						help='Momentum for Adam Update')

	parser.add_argument('--epochs', type=int, default=200,
						help='Max number of epochs')

	parser.add_argument('--save_every', type=int, default=50,
						help='Save Model/Samples every x iterations over '
							 'batches')

	parser.add_argument('--resume_model', type=bool, default=False,
						help='Pre-Trained Model load or not')

	parser.add_argument('--data_set', type=str, default="mnist",
						help='Dat set: mnist, flowers')

	parser.add_argument('--model_name', type=str, default="AutoGANDoc",
						help='model_1 or model_2')

	parser.add_argument('--train', type=bool, default=True,
						help='True while training and otherwise')

	parser.add_argument('--validate_every', type=int, default=1,
						help='run validation after how many epochs')

	args = parser.parse_args()

	main(args)
