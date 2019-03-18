import numpy as np
from outputLayer import OutputLayer
from hiddenLayer import HiddenLayer
from Model import Model
from keras.datasets import cifar10, mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import urllib
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(data):
	
	values = np.array(data)
	
	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

	return onehot_encoded

def load_iris_data():
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	raw_data = urllib.urlopen(url)
	dataset = np.genfromtxt(raw_data, dtype = 'str', missing_values='?', delimiter=',')

	y_train = one_hot_encode(dataset[:,-1])
	x_train = dataset[:,:-1].astype(np.float32)
	x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state = 42)

  	return (x_train, y_train), (x_test, y_test)


def load_mnist_data():
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images = train_images.reshape(60000, (28*28))
	train_images = train_images.astype('float128') / 255

	test_images = test_images.reshape(10000, (28*28))
	test_images = test_images.astype('float128') / 255
	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)

	return (train_images, train_labels), (test_images, test_labels)

def plot_metrics(metrics, epochs, file_name = 'metrics.png'):

	file_name = "../results/" + file_name
	print(file_name)
	accuracy = metrics['acc']
	val_accuracy = metrics['val_acc']
	loss = metrics['loss']
	val_loss = metrics['val_loss']
	fig = plt.figure(figsize=(15,7))

	plt.subplot(1,2,1)
	plt.plot(range(epochs), accuracy, 'r', label = 'Train Accuracy')
	plt.plot(range(epochs), val_accuracy, 'b', label = 'Validation Accuracy')
	plt.legend()
	  
	plt.subplot(1,2,2)
	plt.plot(range(epochs), loss, 'r', label = 'Train Loss')
	plt.plot(range(epochs), val_loss, 'b', label = 'Validation Loss')

	plt.legend()
	plt.savefig(file_name)

def load_cifar_subset(num_classes = 3):
  (x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()
  
  num_features = x_train_all.shape[1] * x_train_all.shape[2] * x_train_all.shape[3]
  
  x_train_all = x_train_all.reshape(-1, num_features)
  x_test_all = x_test_all.reshape(-1, num_features)

  train_keep = (y_train_all < num_classes).reshape(-1)
  test_keep = (y_test_all < num_classes).reshape(-1)

  y_train = y_train_all[train_keep,]
  x_train = x_train_all[train_keep,]
  y_test = y_test_all[test_keep,]
  x_test = x_test_all[test_keep,]

  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255

  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  return (x_train, y_train), (x_test, y_test)

def train(x_train, y_train, x_val = 0, y_val = 0):
	model = Model()
	model.build_model(input_size = x_train.shape[-1], output_size = y_train.shape[-1], hidden_layer_info = hidden_layers)
	metrics = model.fit(x_train = x_train, y_train = y_train, x_val = x_test, y_val = y_test, learning_rate = learning_rate, epochs = epochs)
	return metrics


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument("--learning_rate", type = float, default = 0.01, help="Learning rate")
	parser.add_argument("--epochs", type = int, default = 30, help="No. of epochs")
	parser.add_argument("--hidden_layers", type = str, default = [256], help="No. of epochs")
	args = parser.parse_args()

	epochs = args.epochs
	learning_rate = args.learning_rate
	hidden_layers = args.hidden_layers
	hidden_layers = map(int, hidden_layers.strip('[]').split(','))

	##Train on IRIS Dataset, with a subset of 3 classes

	(x_train, y_train), (x_test, y_test) = load_iris_data()

	metrics = train(x_train, y_train, x_test, y_test)
	plot_metrics(metrics, epochs, 'iris_metrics.png')
	
	##Train on MNIST Dataset
	(x_train, y_train), (x_test, y_test) = load_mnist_data()
	
	x_train =x_train[1:10000]
	y_train = y_train[1:10000]
	

	metrics = train(x_train, y_train, x_test, y_test)
	plot_metrics(metrics, epochs, 'mnist_metrics.png')

	


	
	

