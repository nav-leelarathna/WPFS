
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
import torch
from load_datasets import sample_metabric, sample_metabric_dataset_vary_sizes, get_metabric, load
from sklearn.model_selection import train_test_split
import utils
import argparse
import numpy as np
import copy 
from torchnmf.nmf import NMF
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import json
from sklearn.cluster import KMeans
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
	def __init__(self, tensor_x, tensor_y , vae, dataset_name, splits) -> None:
		self.tensor = tensor_x
		self.tensor_y = tensor_y
		self.vae = vae

		s = copy.deepcopy(splits)
		for i in range(1, len(splits)):
			s[i] += s[i-1]
		for i in range(len(splits)):
			s[i] = int(s[i] * self.tensor.shape[1] / s[-1])
		s = s[:-1]
		self.arrays = np.split(tensor_x, indices_or_sections=s, axis=1)

		if vae is not None:
			self.arrays = [torch.from_numpy(a).to(device) for a in self.arrays]
			self.arrays = vae.get_latent(self.arrays).detach().to(torch.device("cpu"))
			print(f"Dataset size: {self.arrays.shape}")
			# print(f"dataset is on {self.arrays.device}")

	def splits(self):
		if self.vae is None:
			return [a.shape[1] for a in self.arrays]
		else:
			return [self.arrays.shape[1]]
		

	def __len__(self):
		return self.tensor.shape[0]

	def __getitem__(self, idx):
		# currently (number of samples, vector)
		if torch.is_tensor(idx):
			idx = idx.tolist()
		if self.vae is None:
			x = [arr[idx,:] for arr in self.arrays]
		else:
			x = self.arrays[idx,:]
		sample = {"x" : x, "y" : self.tensor_y[idx]}
		# returns a list of numpy arrays
		return sample
	
class OverlappingFeaturesDataset(Dataset):
	def __init__(self, tensor_x, tensor_y , vae, dataset_name, k, percentage_overlap) -> None:
		self.tensor = tensor_x
		num_features = tensor_x.shape[1]
		self.tensor_y = tensor_y
		self.vae = vae

		subset_start_stop_indices = utils.generate_overlapping_sets(num_features=num_features, subsets=k, percentage_overlap=percentage_overlap)
		indices =np.arange(num_features)
		# np.random.shuffle(indices)
		self.subset_indices = [indices[start_stop[0]:start_stop[1]] for start_stop in subset_start_stop_indices]
		# TODO save these indices for later
		self.arrays = [tensor_x[:,s_i] for s_i in self.subset_indices]

		if vae is not None:
			self.arrays = [torch.from_numpy(a).to(device) for a in self.arrays]
			self.arrays = vae.get_latent(self.arrays).detach().to(torch.device("cpu"))
			print(f"Dataset size: {self.arrays.shape}")
			# print(f"dataset is on {self.arrays.device}")

	def splits(self):
		if self.vae is None:
			return [a.shape[1] for a in self.arrays]
		else:
			return [self.arrays.shape[1]]
		
	def __len__(self):
		return self.tensor.shape[0]

	def __getitem__(self, idx):
		# currently (number of samples, vector)
		if torch.is_tensor(idx):
			idx = idx.tolist()
		if self.vae is None:
			x = [arr[idx,:] for arr in self.arrays]
		else:
			x = self.arrays[idx,:]
		sample = {"x" : x, "y" : self.tensor_y[idx]}
		# returns a list of numpy arrays
		return sample

class KMeansDataset(Dataset):
	def __init__(self, tensor_x, tensor_y, vae, dataset_name, k):
		indices = load_kmeans_indices(dataset_name, k)
		self.tensor_x = tensor_x# .to_numpy()
		self.tensor_y = tensor_y
		self.vae = vae
		# print(self.tensor_x.shape)
		assert len(indices) == self.tensor_x.shape[1]
		self.arrays = [[] for _ in range(k)]
		for i, modality_index in enumerate(indices):
			self.arrays[modality_index].append(self.tensor_x[:, i])
		self.arrays = [np.stack(arr, axis=-1) for arr in self.arrays]
		# for arr in self.arrays:
		#     print(arr.shape)
		# print(sum([s.shape[1] for s in self.arrays]))
		if vae is not None:
			self.arrays = [torch.from_numpy(a) for a in self.arrays]
			self.arrays = vae.get_latent(self.arrays).detach().to(torch.device("cpu"))
			print(f"Dataset size: {self.arrays.shape}")

	def splits(self):
		if self.vae is None:
			return [a.shape[1] for a in self.arrays]
		else:
			return [self.arrays.shape[1]]

	def __len__(self):
		return self.tensor_x.shape[0]

	def __getitem__(self, idx):
		if self.vae is None:
			x = [arr[idx, :] for arr in self.arrays]
		else:
			 x = self.arrays[idx,:]
		sample = {"x" : x, "y": self.tensor_y[idx]}
		return sample
	
class PatchedDataset(Dataset):
	# Treats the data as if were once a 2d image and uses patches as the subsets
	def __init__(self, tensor_x, tensor_y, vae, dataset_name, k):
		# print(self.tensor_x.shape)
		self.tensor_x = tensor_x
		self.vae = vae
		self.tensor_y = tensor_y
		num_samples = tensor_x.shape[0]
		tensor_x.shape = (num_samples, 28,28)
		num_patches_h = num_patches_w = int(k**0.5)

		arr = np.array_split(ary=tensor_x,indices_or_sections=num_patches_h,axis=1)
		arr = utils.flatten([np.array_split(ary=a,indices_or_sections=num_patches_w,axis=2) for a in arr])
		self.arrays =[a.reshape((num_samples,-1)) for a in arr]

		if vae is not None:
			self.arrays = [torch.from_numpy(a) for a in self.arrays]
			self.arrays = vae.get_latent(self.arrays).detach().to(torch.device("cpu"))
			print(f"Dataset size: {self.arrays.shape}")
		
	def splits(self):
		if self.vae is None:
			return [a.shape[1] for a in self.arrays]
		else:
			return [self.arrays.shape[1]]

	def __len__(self):
		return self.tensor_x.shape[0]

	def __getitem__(self, idx):
		if self.vae is None:
			x = [arr[idx, :] for arr in self.arrays]
		else:
			 x = self.arrays[idx,:]
		sample = {"x" : x, "y": self.tensor_y[idx]}
		return sample

def compute_histogram_embedding(X, embedding_size):
	"""
	Compute embedding_matrix (D x M) based on the histograms. The function implements two methods:

	DietNetwork
	- Normalized bincounts for each SNP

	FsNet
	0. Input matrix NxD
	1. Z-score standardize each column (mean 0, std 1)
	2. Compute the histogram for every feature (with density = False)
	3. Multiply the histogram values with the bin mean

	:param (N x D) X: dataset, each row representing one sample
	:return np.ndarray (D x M) embedding_matrix: matrix where each row represents the embedding of one feature
	"""
	X = np.rot90(X)
	
	number_features = X.shape[0]
	embedding_matrix = np.zeros(shape=(number_features, embedding_size))

	for feature_id in range(number_features):
		feature = X[feature_id]

		hist_values, bin_edges = np.histogram(feature, bins=embedding_size) # like in FsNet
		bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
		embedding_matrix[feature_id] = np.multiply(hist_values, bin_centers)

	return embedding_matrix


def compute_nmf_embeddings(Xt, rank):
	"""
	Note: torchnmf computes V = H W^T instead of the standard formula V = W H

	Input
	- V (D x N)
	- rank of NMF

	Returns
	- H (D x r) (torch.Parameter with requires_grad=True), where each row represents one gene embedding 
	"""
	print("Approximating V = H W.T")
	print(f"Input V has shape {Xt.shape}")
	assert type(Xt)==torch.Tensor
	assert Xt.shape[0] > Xt.shape[1]

	nmf = NMF(Xt.shape, rank=rank).to(device)
	nmf.fit(Xt.to(device), beta=2, max_iter=1000, verbose=True) # beta=2 coresponds to the Frobenius norm, which is equivalent to an additive Gaussian noise model

	print(f"H has shape {nmf.H.shape}")
	print(f"W.T has shape {nmf.W.T.shape}")

	return nmf.H, nmf.W


def compute_svd_embeddings(X, rank=None):
	"""
	- X (N x D)
	- rank (int): rank of the approximation (i.e., size of the embedding)
	"""
	assert type(X)==torch.Tensor
	assert X.shape[0] < X.shape[1]

	U, S, Vh = torch.linalg.svd(X, full_matrices=False)

	V = Vh.T

	if rank:
		S = S[:rank]
		V = V[:rank]

	return V, S

class CustomDataModule(pl.LightningDataModule):
	def __init__(self, configuration, X_train, y_train, X_valid, y_valid, X_test, y_test):
		super().__init__()
		self.configuration = configuration
		self.batch_size = configuration.data_module.batch_size

		# Standardize data
		self.X_train = X_train.astype(np.float32)
		self.X_valid = X_valid.astype(np.float32)
		self.X_test = X_test.astype(np.float32)

		self.vae = None
		if configuration.data_module.latent.enabled:
			print("Using latent representation of dataset")
			# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			self.vae = utils.load_model_from_wandb(**(configuration.data_module.latent.args.__dict__)).to(device)
			print(f"vae device: {self.vae.device}")

		######
		# PERFORM PCA OR NMF TRANSFORM HERE, make it a config option
		######
		if hasattr(configuration.data_module, "embedding"):
			if configuration.data_module.embedding.enabled:
				print(f"Performing embedding using {configuration.data_module.embedding.name[1]}")
				embedder = utils.init_obj(configuration.data_module.embedding)
				embedder.fit(self.X_train)
				self.X_train = embedder.transform(self.X_train)
				self.X_valid = embedder.transform(self.X_valid)
				self.X_test = embedder.transform(self.X_test)

		self.y_train = y_train.astype(np.int64)
		self.y_valid = y_valid.astype(np.int64)
		self.y_test = y_test.astype(np.int64)

		self.configuration.data_module.train_size = X_train.shape[0]
		self.configuration.data_module.valid_size = X_valid.shape[0]
		self.configuration.data_module.test_size = X_test.shape[0]

	def prepare_data(self):
		pass

	def setup(self, stage=None):
		print(self.configuration.data_module.dataset.name)
		self.train = utils.init_obj(self.configuration.data_module.dataset, self.X_train, self.y_train, self.vae, self.configuration.data_module.name, init_type="Dataset")
		self.valid = utils.init_obj(self.configuration.data_module.dataset, self.X_valid, self.y_valid, self.vae, self.configuration.data_module.name,init_type="Dataset")
		self.test = utils.init_obj(self.configuration.data_module.dataset, self.X_test, self.y_test, self.vae, self.configuration.data_module.name,init_type="Dataset")

	def splits(self):
		return self.train.splits()

	def train_dataloader(self):
		return DataLoader(self.train, self.batch_size, num_workers=8, pin_memory=True)
		# Return DataLoader for Training Data here
	
	def val_dataloader(self):
		return DataLoader(self.valid, self.batch_size, num_workers=8, pin_memory=True)
		# Return DataLoader for Validation Data here
	
	def test_dataloader(self):
		return DataLoader(self.test, self.batch_size, num_workers=8, pin_memory=True)
	
	def get_embedding_matrix(self, embedding_type, embedding_size):
		"""
		Return matrix D x M

		Use a the shared hyper-parameter self.args.embedding_preprocessing.
		"""
		if embedding_type == None:
			return None
		else:
			if embedding_size == None:
				raise Exception()

		# Preprocess the data for the embeddings
		# if self.args.embedding_preprocessing == 'raw':
		# 	X_for_embeddings = self.X_train_raw
		# elif self.args.embedding_preprocessing == 'z_score':
		# 	X_for_embeddings = StandardScaler().fit_transform(self.X_train_raw)
		# elif self.args.embedding_preprocessing == 'minmax':
		# 	X_for_embeddings = MinMaxScaler().fit_transform(self.X_train_raw)
		# else:
		# 	raise Exception("embedding_preprocessing not supported")
		# X_for_embeddings = MinMaxScaler().fit_transform(self.X_train)
		X_for_embeddings = self.X_train

		if embedding_type == 'histogram':
			"""
			Embedding similar to FsNet
			"""
			embedding_matrix = compute_histogram_embedding(X_for_embeddings, embedding_size)
			return torch.tensor(embedding_matrix.copy(), dtype=torch.float32, requires_grad=False)
		elif embedding_type=='feature_values':
			"""
			A gene's embedding are its patients gene expressions.
			"""
			embedding_matrix = np.rot90(X_for_embeddings)[:, :embedding_size]
			return torch.tensor(embedding_matrix.copy(), dtype=torch.float32, requires_grad=False)
		elif embedding_type=='svd':
			# Vh.T (4160 x rank) contains the gene embeddings on each row
			U, S, Vh = torch.linalg.svd(torch.tensor(X_for_embeddings, dtype=torch.float32), full_matrices=False) 
			
			Vh.T.requires_grad = False
			return Vh.T[:, :embedding_size].type(torch.float32)
		elif embedding_type=='nmf':
			H, _ = compute_nmf_embeddings(torch.tensor(X_for_embeddings).T, rank=embedding_size)
			H_data = H.data
			H_data.requires_grad = False
			return H_data.type(torch.float32)
		else:
			raise Exception("Invalid embedding type")


def compute_stratified_splits(X, y, cv_folds, seed_kfold, split_id):
	skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed_kfold)
	for i, (train_ids, test_ids) in enumerate(skf.split(X, y)):
		if i == split_id:
			return X[train_ids], X[test_ids], y[train_ids], y[test_ids]
				
def compute_reduced_dataset(X, y, seed, trainAndValidProportion):
	numFeatures = X.shape[0]
	indices = list(range(numFeatures))
	random.Random(seed).shuffle(indices)
	randomisedIndices = np.array(indices)
	trainAndValidIndices = randomisedIndices[:int(trainAndValidProportion*numFeatures)]
	testIndices = randomisedIndices[int(trainAndValidProportion*numFeatures):]
	return X[trainAndValidIndices], X[testIndices], y[trainAndValidIndices], y[testIndices]


def create_datamodule_with_cross_validation(configuration, X, y, args):
	"""
	Split X, y to be suitable for k-fold cross-validation.
	It uses args.test_split to create the train, valid and test stratified datasets.
	"""
	if type(X)==pd.DataFrame:
		X = X.to_numpy()
	if type(y)==pd.Series:
		y = y.to_numpy()

	# shuffle samples
	num_samples = X.shape[0]
	sampleIndices = list(range(num_samples))
	random.Random(configuration.data_module.seed_kfold).shuffle(sampleIndices)
	X = X[np.array(sampleIndices),:]
	y = y[np.array(sampleIndices)]
	
	# shuffle X and y 
	num_features = X.shape[1]
	shuffledIndices = list(range(num_features))
	if not hasattr(configuration.data_module, "shuffle_features"):
		# shuffle the features by default unless turned off
		configuration.data_module.shuffle_features = True

	if configuration.data_module.shuffle_features:
		print("Shuffling order of features in dataset")
		random.Random(configuration.data_module.seed_kfold).shuffle(shuffledIndices)
		X = X[:, np.array(shuffledIndices)]
	else:
		print("Not shuffling order of features in dataset")
	
	if not hasattr(configuration.data_module, "dataset_percentage"):
		configuration.data_module.dataset_percentage = 1

	X_train_and_valid, X_test, y_train_and_valid, y_test = compute_stratified_splits(
	X, y, cv_folds=configuration.data_module.cv_folds, seed_kfold=configuration.data_module.seed_kfold, split_id=configuration.data_module.split_id)
		
	print(f"Using {(100*configuration.data_module.dataset_percentage):.3f}% of available test set")
	if configuration.data_module.dataset_percentage < 1:
		numSamples = X_train_and_valid.shape[0]
		reducedNumSamples = int(configuration.data_module.dataset_percentage * numSamples)
		X_train_and_valid, y_train_and_valid =  X_train_and_valid[:reducedNumSamples], y_train_and_valid[:reducedNumSamples]
	# randomly pick a validation set from the training_and_val data
	if X_train_and_valid.shape[0] < 300:
		configuration.data_module.valid_percentage = 0.25
	X_train, X_valid, y_train, y_valid = train_test_split(
		X_train_and_valid, y_train_and_valid,
		test_size = configuration.data_module.valid_percentage,
		random_state = configuration.data_module.seed_validation,
		stratify = y_train_and_valid
	)
	print(f"Size of train : {X_train.shape}")
	print(f"Size of valid: {X_valid.shape}")
	print(f"Size of test : {X_test.shape}")
	if args is not None:
		args.train_size = X_train.shape[0]
		args.valid_size = X_valid.shape[0]
		args.test_size = X_test.shape[0]

	# assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == X.shape[0]
	# assert set(y_train).union(set(y_valid)).union(set(y_test)) == set(y)
	if configuration.data_module.class_weight_type=='balanced':
		class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
	elif configuration.data_module.class_weight_type=='standard':
		class_weights = compute_class_weight(class_weight=None, classes=np.unique(y), y=y)
	return CustomDataModule(configuration, X_train, y_train, X_valid, y_valid, X_test, y_test), class_weights

def create_datamodule(configuration, args):
	print("Training model on " + configuration.data_module.name)
	if configuration.debug:
		X = np.random.normal(size=(1977,4160))
		y = np.ones((1977,))
	else:
		X, y = load(configuration.data_module.name) 
	if args is not None:
		args.num_features = X.shape[1]
	data_module, class_weights = create_datamodule_with_cross_validation(configuration, X, y, args)
	data_module.prepare_data()
	data_module.setup()
	print("Size of modalities: " + ", ".join([str(s) for s in data_module.splits()]))
	
	configuration.data_module.class_weights = class_weights.astype(np.float32).tolist()
	input_sizes = data_module.splits()
	configuration.data_module.modality_widths = input_sizes

	configuration.model.args.input_sizes = input_sizes
	configuration.model.args.class_weights = configuration.data_module.class_weights 
	return data_module

def k_means():
	name = "lung"
	# X, y = load(configuration.data_module.name)
	X, y = load(name)
	print(type(X))
	X = X.T
	print(X.shape)
	kmeans = KMeans(
	init="random",
	n_clusters=3,
	n_init=10,
	max_iter=300,
	random_state=42)
	kmeans.fit(X)
	print(kmeans.labels_)

def generate_kmeans_indices(seed=42):
	datasets = ["lung", "metabric-dr", "metabric-pam50"]
	modality_indices = {}
	for dataset in datasets:
		X, _ = load(dataset)
		X = X.T
		dataset_indices = {}
		dataset_indices[1] = [0 for _ in range(X.shape[0])]
		for k in range(2,9):
			kmeans = KMeans(
			init="random",
			n_clusters=k,
			n_init=10,
			max_iter=10000,
			random_state=seed)
			kmeans.fit(X)
			dataset_indices[k] = kmeans.labels_.tolist()
			print(f"Computing indices for {dataset}, k={k}")
		modality_indices[dataset] = dataset_indices
	json.dump(modality_indices, open("gene_data/k_means_indices.txt",'w'))

def load_kmeans_indices(dataset_name, k):
	assert dataset_name in ["lung", "metabric-dr", "metabric-pam50"]
	return json.load(open("gene_data/k_means_indices.txt"))[dataset_name][str(k)]

def test_kmeans(configuration):
	name = "lung"
	k = 2
	X,y = load(name)
	# vae = utils.load_model_from_wandb(**(configuration.data_module.latent.args.__dict__))
	# dataset = KMeansDataset(X, y, vae, name, k)
	dataset = KMeansDataset(X.to_numpy(), y, None, name, k)
	x = dataset[28]
	print(x)
	print(len(dataset))
	print(dataset.splits())

if __name__ == "__main__":
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-c', '--config', type=str, default='configs/classifier_metabric_dr.json', help='JSON file for configuration')
	# args = parser.parse_args()
	# configuration = utils.parse(args.config)
	# test_kmeans(configuration)
	generate_kmeans_indices()
	