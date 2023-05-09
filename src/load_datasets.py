import scipy.io as spio
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn import preprocessing
import torch
# from torchvision.transforms import ToTensor
from sklearn.utils.multiclass import type_of_target
# from torchvision import datasets
import rdata
DATA_DIR = 'data' # TODO: set the path to the data directory 

def load_lung_gordon():
	# Gordon lung cancer dataset
	path = f"{DATA_DIR}/gordon/gordon.RData"
	data = rdata.parser.parse_file(path)
	data = rdata.conversion.convert(data)
	label_dict = {
		'adenocarcinoma': 0,
        'mesothelioma': 1,
	}
	X = data['gordon']['x'].to_numpy()
	y = pd.Series(data['gordon']['y'].to_numpy())
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(X)
	X = pd.DataFrame(x_scaled)
	y = y.replace(label_dict)
	return X, y

def load_breast_cancer():
	# Gordon lung cancer dataset
	path = f"{DATA_DIR}/chowdary/chowdary.RData"
	data = rdata.parser.parse_file(path)
	data = rdata.conversion.convert(data)
	classes = ['breast' ,'colon']
	label_dict = {
		'breast': 0,
        'colon': 1,
	}
	X = data['chowdary']['x'].to_numpy()
	y = pd.Series(data['chowdary']['y'].to_numpy())
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(X)
	X = pd.DataFrame(x_scaled)
	y = y.replace(label_dict)
	return X, y

def load_blog():
	parent_dict = f"{DATA_DIR}/BlogFeedback/"
	files = os.listdir(parent_dict)
	data = []
	for file in files:
		path = os.path.join(parent_dict, file)
		arr = np.loadtxt(path,
                 delimiter=",", dtype=float)
		data.append(arr)
	dataset = np.concatenate(data)
	features = dataset.shape[1]
	X = dataset[:,:features-1]
	y = dataset[:, features-1]
	y[y>0] = 1
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(X)
	X = pd.DataFrame(x_scaled)
	return X, y

def load_income():
       
	x_train = np.load(f"{DATA_DIR}/income/train_feat_std.npy")
	y_train = np.load(f"{DATA_DIR}/income/train_label.npy")
	x_test = np.load(f"{DATA_DIR}/income/test_feat_std.npy")
	y_test = np.load(f"{DATA_DIR}/income/test_label.npy")
	X = np.concatenate([x_train, x_test])	
	y = np.concatenate([y_train, y_test])	
	min_max_scaler = preprocessing.MinMaxScaler()
	X = min_max_scaler.fit_transform(X)
	# return x_train, y_train, x_test, y_test
	return X, y

# def load_mnist():
# 	train = datasets.MNIST(root=DATA_DIR, train=True,download=True,transform=ToTensor())
# 	test = datasets.MNIST(root=DATA_DIR, train=False,download=True,transform=ToTensor())
# 	train_x, train_y = train.data.numpy(), train.targets.numpy()
# 	test_x, test_y = test.data.numpy(), test.targets.numpy()
# 	X = np.concatenate([train_x,test_x],axis=0)
# 	y = np.concatenate([train_y,test_y],axis=0)
# 	# flatten and normalise X
# 	# X = np.flatten(X, start_dim=1) / 255
# 	num_samples = X.shape[0]
# 	X.shape = (num_samples, -1)
# 	X = X / 255.
# 	# print(X.shape)
# 	# print(y.shape)
# 	# print(X)
# 	# print(X.max())
# 	# print(X.min())
# 	# print(y)
# 	return X, y

def load_obesity():
	# read file
	dtype = 'float64'
	identifier_string = "gi|"
	identifier_string = "k__"
	label_string = 'disease'
	
	# filename = DATA_DIR + "/obesity/marker_Obesity.txt"
	filename = DATA_DIR + "/abundance/abundance_Obesity.txt"
	if os.path.isfile(filename):
		raw = pd.read_csv(filename, sep='\t', index_col=0, header=None)
	else:
		print("FileNotFoundError: File {} does not exist".format(filename))
		exit()
	# select rows having feature index identifier string
	X = raw.loc[raw.index.str.contains(identifier_string, regex=False)].T
	label_dict = {
		'n': 0,
        # T2D and WT2D
        't2d': 1,
		'leaness': 0, 'obesity': 1,
	}
	# get class labels
	Y = raw.loc[label_string] #'disease'
	Y = Y.replace(label_dict)
	# train and test split
	# X_train, X_test, y_train, y_test = train_test_split(X.values.astype(dtype), Y.values.astype('int'), test_size=0.2, random_state=self.seed, stratify=Y.values)
	# print(X.shape)
	# print(Y.shape)
	# print(Y.sum(0))
	# X = (X-X.min())/(X.max()-X.min())
	X = X.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(X)
	X = pd.DataFrame(x_scaled)
	return X, Y
	
def load_t2d():
	# read file
	dtype = 'float64'
	identifier_string = "gi|"
	identifier_string = "k__"
	label_string = 'disease'
	# filename = DATA_DIR + "/t2d/marker_T2D.txt"
	filename = DATA_DIR + "/abundance/abundance_T2D.txt"
	if os.path.isfile(filename):
		raw = pd.read_csv(filename, sep='\t', index_col=0, header=None)
	else:
		print("FileNotFoundError: File {} does not exist".format(filename))
		exit()
	# select rows having feature index identifier string
	X = raw.loc[raw.index.str.contains(identifier_string, regex=False)].T
	label_dict = {
		'n': 0,
        # T2D and WT2D
        't2d': 1,
		'leaness': 0, 'obesity': 1,
	}
	# get class labels
	Y = raw.loc[label_string] #'disease'
	Y = Y.replace(label_dict)
	# train and test split
	# X_train, X_test, y_train, y_test = train_test_split(X.values.astype(dtype), Y.values.astype('int'), test_size=0.2, random_state=self.seed, stratify=Y.values)
	# print(X.shape)
	# print(Y.shape)
	# print(Y.sum(0))
	X = X.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(X)
	X = pd.DataFrame(x_scaled)
	return X, Y

################### Load datasets ###################
def load_lung(drop_class_5=True):
	"""
	Labels in initial dataset:
	1    139
	2     17
	3     21
	4     20
	5      6

	We drop the class 5 because it has too little examples.
	"""
	data = spio.loadmat(f'{DATA_DIR}/lung.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	if drop_class_5:
		# Examples of class 5 are deleted
		X = X.drop(index=[156, 157, 158, 159, 160, 161])
		Y = Y.drop([156, 157, 158, 159, 160, 161])

	new_labels = {1:0, 2:1, 3:2, 4:3, 5:4}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_prostate():
	""""
	Labels in initial dataset:
	1    50
	2    52
	"""
	data = spio.loadmat(f'{DATA_DIR}/Prostate_GE.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1}
	Y = Y.apply(lambda x: new_labels[x])

	return X, Y

def load_toxicity():
	"""
	Labels in initial dataset:
	1    45
	2    45
	3    39
	4    42
	"""
	data = spio.loadmat(f'{DATA_DIR}/TOX_171.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1, 3:2, 4:3}
	Y = Y.apply(lambda x: new_labels[x])
	X = X.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(X)
	X = pd.DataFrame(x_scaled)
	return X, Y

def load_cll():
	"""
	Labels in initial dataset:
	1    11
	2    49
	3    51
	"""
	data = spio.loadmat(f'{DATA_DIR}/CLL_SUB_111.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1, 3:2}
	Y = Y.apply(lambda x: new_labels[x])
	X = X.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(X)
	X = pd.DataFrame(x_scaled)
	return X, Y

def load_smk():
	"""
	Labels in initial dataset:
	1    90
	2    97
	"""
	data = spio.loadmat(f'{DATA_DIR}/SMK_CAN_187.mat')
	X = pd.DataFrame(data['X'])
	Y = pd.Series(data['Y'][:, 0])

	new_labels = {1:0, 2:1}
	Y = Y.apply(lambda x: new_labels[x])
	X = X.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(X)
	X = pd.DataFrame(x_scaled)
	return X, Y


def get_tcga(dataset_name):
	"""
	Create dataset splits from TCGA

	- dataset_name: name of the dataset (e.g. 'tcga-tumor-grade', 'tcga-2ysurvival')
		- 'tcga-tumor-grade': dataset with tumor grade labels
		- 'tcga-2ysurvival': dataset with 2-year survival labels
	- dataset_size: number of samples to take from the dataset
	- random_state: random seed for the sampling
	"""
	assert dataset_name in ['tcga-tumor-grade', 'tcga-2ysurvival']

	data_folder = f'{DATA_DIR}'

	tcga_full = pd.read_csv(f'{data_folder}/TCGA_full/tcga_hncs.csv', index_col=0)
	tcga_full = tcga_full.dropna()

	# filter genes based on the HALLMARK gene set
	partner_genes_to_filter = pd.read_csv(f'{data_folder}/imp_genes_list.csv',index_col=0)
	set_partner_genes_to_filter = set(partner_genes_to_filter.index)
	set_partner_genes_to_filter

	# Clean the set of columns
	column_names_clean = []
	for column_with_number in tcga_full.columns:
		column_name = column_with_number.split('|')[0]
		column_names_clean.append(column_name)

	# keep only the HALLMARK genes
	tcga_full_columns_changed = tcga_full.copy()
	tcga_full_columns_changed.columns = column_names_clean
	genes_intersection = list(set(column_names_clean).intersection(set_partner_genes_to_filter))
	tcga_only_intersection_genes = tcga_full_columns_changed[genes_intersection]

	if dataset_name == 'tcga-tumor-grade':
		tumor_grade = tcga_full['tumor_grade']

		# Keep only G1, G2, G3 classes
		tcga_genes_and_tumor_grade = tcga_only_intersection_genes.copy().merge(tumor_grade, left_index=True, right_index=True, validate='one_to_one')
		tcga_genes_and_tumor_grade = tcga_genes_and_tumor_grade.loc[tcga_genes_and_tumor_grade['tumor_grade'].isin(['G1', 'G2', 'G3'])]
		tcga_genes_and_tumor_grade['tumor_grade'] = tcga_genes_and_tumor_grade['tumor_grade'].map({'G1': int(0), 'G2': int(1), 'G3': int(2)})

		# dataset, _ = train_test_split(tcga_genes_and_tumor_grade, train_size=dataset_size,
		# 	stratify=tcga_genes_and_tumor_grade['tumor_grade'], random_state=random_state)
		# return dataset
		X = tcga_genes_and_tumor_grade[tcga_genes_and_tumor_grade.columns[:-1]].to_numpy()
		y = tcga_genes_and_tumor_grade[tcga_genes_and_tumor_grade.columns[-1]].to_numpy()
		# X = X.values #returns a numpy array
		min_max_scaler = preprocessing.MinMaxScaler()
		X = min_max_scaler.fit_transform(X)
		# X = pd.DataFrame(x_scaled)
		return X, y

	else:
		two_year_survival = tcga_full['X2yr.RF.Surv.']

		tcga_genes_and_2ysurvival = tcga_only_intersection_genes.copy().merge(two_year_survival, left_index=True, right_index=True, validate='one_to_one')

		# dataset, _ = train_test_split(tcga_genes_and_2ysurvival, train_size=dataset_size,
		# 	stratify=tcga_genes_and_2ysurvival['X2yr.RF.Surv.'], random_state=random_state)
		X = tcga_genes_and_2ysurvival[tcga_genes_and_2ysurvival.columns[:-1]].to_numpy()
		y = tcga_genes_and_2ysurvival[tcga_genes_and_2ysurvival.columns[-1]].to_numpy()
		min_max_scaler = preprocessing.MinMaxScaler()
		X = min_max_scaler.fit_transform(X)
		return X,y

		# return dataset


def get_metabric(dataset_name):
	"""
	Create dataset splits from Metabric

	- dataset_name: name of the dataset (e.g. 'metabric-dr', 'metabric-pam50')
		- 'metabric-dr': dataset with drug response labels
		- 'metabric-pam50': dataset with PAM50 labels
	- dataset_size: number of samples to take from the dataset
	- random_state: random seed for the sampling
	"""
	assert dataset_name in ['metabric-dr', 'metabric-pam50']

	data_folder = DATA_DIR

	# load expression data
	expressionsMB = pd.read_csv(f'{data_folder}/Metabric_full/MOLECULARDATA/CURTIS_data_Expression.txt', delimiter='\t').T
	expressionsMB.columns = expressionsMB.iloc[0]
	expressionsMB.drop(expressionsMB.index[[0,1]],inplace=True)
	expressionsMB_genes = expressionsMB.T.copy()

	# load Hallmark gene set
	genes_to_filter = pd.read_csv(f'{data_folder}/imp_genes_list.csv',index_col=0)
	genes_to_filter_unduplicated = genes_to_filter.loc[~genes_to_filter.index.duplicated(keep='first')]

	# keep only the genes from Hallmark
	expressionsMB_filtered = pd.concat([genes_to_filter_unduplicated, expressionsMB_genes],axis=1, join="inner").copy()
	# Transpose to have samples per rows, and drop null rows
	expressionsMB_filtered = expressionsMB_filtered.T.copy().dropna()

	# load clinical data
	clinMB=pd.read_csv(f'{data_folder}/Metabric_full/MOLECULARDATA/TableS6.txt', delimiter='\t')
	clinMB.set_index('METABRIC.ID',inplace=True)

	if dataset_name == 'metabric-dr':
		DR = clinMB['DR'].copy().dropna()
		expressions_with_DR = expressionsMB_filtered.merge(DR, left_index=True, right_index=True, validate='one_to_one')
		
		# dataset, _ = train_test_split(expressions_with_DR, train_size=dataset_size, 
			# stratify=expressions_with_DR['DR'], random_state=random_state)
		return expressions_with_DR[expressions_with_DR.columns[:-1]].to_numpy(), expressions_with_DR[expressions_with_DR.columns[-1]].to_numpy()

	elif dataset_name == 'metabric-pam50':
		pam50 = clinMB['Pam50Subtype'].copy().dropna()
		# convert to binary labels Basal vs. non-Basal
		name_to_value = {
			'Basal': int(1), 
			'LumA': int(0),
			'LumB': int(0),
			'Her2': int(0),
			'Normal': int(0)
		}
		pam50_binary = pam50.map(name_to_value).astype(int)
		expressions_with_pam50 = expressionsMB_filtered.merge(pam50_binary, left_index=True, right_index=True, validate='one_to_one')

		# dataset, _ = train_test_split(expressions_with_pam50, train_size=dataset_size,
			# stratify=expressions_with_pam50['Pam50Subtype'], # random_state=random_state)

		return expressions_with_pam50[expressions_with_pam50.columns[:-1]].to_numpy(), expressions_with_pam50[expressions_with_pam50.columns[-1]].to_numpy()


def load(dataset):
	if dataset in ['metabric-pam50', 'metabric-dr', 'tcga-2ysurvival', 'tcga-tumor-grade']:
		if dataset in ['metabric-pam50', 'metabric-dr']:
			# dataset_path = os.path.join(DATA_DIR, f'Metabric_samples/metabric_pam50_train_{dataset_size}.csv')
			X, y = get_metabric(dataset)
		else:
			X, y = get_tcga(dataset)

	elif dataset in ['mnist','obesity','t2d','lung', 'toxicity', 'prostate', 'cll', 'smk', "blog", "income", "lung_gordon", "breast"]:
		if dataset=='lung':
			X, y = load_lung()
		elif dataset=='lung_gordon':
			X, y = load_lung_gordon()
		elif dataset=='breast':
			X, y = load_breast_cancer()
		elif dataset=='obesity':
			X, y = load_obesity()
		elif dataset=='t2d':
			X, y = load_t2d()
		elif dataset=='toxicity':
			X, y = load_toxicity()
		elif dataset=='prostate':
			X, y = load_prostate()
		elif dataset=='cll':
			X, y = load_cll()
		elif dataset=='smk':
			X, y = load_smk()
		elif dataset=='mnist':
			X, y = load_mnist()
		elif dataset=='blog':
			X, y = load_blog()		
		elif dataset=='income':
			X, y = load_income()
		else:
			raise Exception(f"{dataset} not recognised")
	return X ,y

################### Sample Metabric/TCGA with fixed dataset size (useful for cross-validation) ###################

def sample_metabric(dataset_name, dataset_size, random_state):
	"""
	Create dataset splits from Metabric

	- dataset_name: name of the dataset (e.g. 'metabric-dr', 'metabric-pam50')
		- 'metabric-dr': dataset with drug response labels
		- 'metabric-pam50': dataset with PAM50 labels
	- dataset_size: number of samples to take from the dataset
	- random_state: random seed for the sampling
	"""
	assert dataset_name in ['metabric-dr', 'metabric-pam50']

	data_folder = DATA_DIR

	# load expression data
	expressionsMB = pd.read_csv(f'{data_folder}/Metabric_full/MOLECULARDATA/CURTIS_data_Expression.txt', delimiter='\t').T
	expressionsMB.columns = expressionsMB.iloc[0]
	expressionsMB.drop(expressionsMB.index[[0,1]],inplace=True)
	expressionsMB_genes = expressionsMB.T.copy()

	# load Hallmark gene set
	genes_to_filter = pd.read_csv(f'{data_folder}/imp_genes_list.csv',index_col=0)
	genes_to_filter_unduplicated = genes_to_filter.loc[~genes_to_filter.index.duplicated(keep='first')]

	# keep only the genes from Hallmark
	expressionsMB_filtered = pd.concat([genes_to_filter_unduplicated, expressionsMB_genes],axis=1, join="inner").copy()
	# Transpose to have samples per rows, and drop null rows
	expressionsMB_filtered = expressionsMB_filtered.T.copy().dropna()

	# load clinical data
	clinMB=pd.read_csv(f'{data_folder}/Metabric_full/MOLECULARDATA/TableS6.txt', delimiter='\t')
	clinMB.set_index('METABRIC.ID',inplace=True)

	if dataset_name == 'metabric-dr':
		DR = clinMB['DR'].copy().dropna()
		expressions_with_DR = expressionsMB_filtered.merge(DR, left_index=True, right_index=True, validate='one_to_one')
		
		dataset, _ = train_test_split(expressions_with_DR, train_size=dataset_size, 
			stratify=expressions_with_DR['DR'], random_state=random_state)
		return dataset

	elif dataset_name == 'metabric-pam50':
		pam50 = clinMB['Pam50Subtype'].copy().dropna()
		# convert to binary labels Basal vs. non-Basal
		name_to_value = {
			'Basal': int(1), 
			'LumA': int(0),
			'LumB': int(0),
			'Her2': int(0),
			'Normal': int(0)
		}
		pam50_binary = pam50.map(name_to_value).astype(int)
		expressions_with_pam50 = expressionsMB_filtered.merge(pam50_binary, left_index=True, right_index=True, validate='one_to_one')

		dataset, _ = train_test_split(expressions_with_pam50, train_size=dataset_size,
			stratify=expressions_with_pam50['Pam50Subtype'], random_state=random_state)

		return dataset

def sample_tcga(dataset_name, dataset_size, random_state):
	"""
	Create dataset splits from TCGA

	- dataset_name: name of the dataset (e.g. 'tcga-tumor-grade', 'tcga-2ysurvival')
		- 'tcga-tumor-grade': dataset with tumor grade labels
		- 'tcga-2ysurvival': dataset with 2-year survival labels
	- dataset_size: number of samples to take from the dataset
	- random_state: random seed for the sampling
	"""
	assert dataset_name in ['tcga-tumor-grade', 'tcga-2ysurvival']

	data_folder = f'{DATA_DIR}/data'

	tcga_full = pd.read_csv(f'{data_folder}/TCGA_full/tcga_hncs.csv', index_col=0)
	tcga_full = tcga_full.dropna()

	# filter genes based on the HALLMARK gene set
	partner_genes_to_filter = pd.read_csv(f'{data_folder}/imp_genes_list.csv',index_col=0)
	set_partner_genes_to_filter = set(partner_genes_to_filter.index)
	set_partner_genes_to_filter

	# Clean the set of columns
	column_names_clean = []
	for column_with_number in tcga_full.columns:
		column_name = column_with_number.split('|')[0]
		column_names_clean.append(column_name)

	# keep only the HALLMARK genes
	tcga_full_columns_changed = tcga_full.copy()
	tcga_full_columns_changed.columns = column_names_clean
	genes_intersection = list(set(column_names_clean).intersection(set_partner_genes_to_filter))
	tcga_only_intersection_genes = tcga_full_columns_changed[genes_intersection]

	if dataset_name == 'tcga-tumor-grade':
		tumor_grade = tcga_full['tumor_grade']

		# Keep only G1, G2, G3 classes
		tcga_genes_and_tumor_grade = tcga_only_intersection_genes.copy().merge(tumor_grade, left_index=True, right_index=True, validate='one_to_one')
		tcga_genes_and_tumor_grade = tcga_genes_and_tumor_grade.loc[tcga_genes_and_tumor_grade['tumor_grade'].isin(['G1', 'G2', 'G3'])]
		tcga_genes_and_tumor_grade['tumor_grade'] = tcga_genes_and_tumor_grade['tumor_grade'].map({'G1': int(0), 'G2': int(1), 'G3': int(2)})

		dataset, _ = train_test_split(tcga_genes_and_tumor_grade, train_size=dataset_size,
			stratify=tcga_genes_and_tumor_grade['tumor_grade'], random_state=random_state)
		return dataset

	else:
		two_year_survival = tcga_full['X2yr.RF.Surv.']

		tcga_genes_and_2ysurvival = tcga_only_intersection_genes.copy().merge(two_year_survival, left_index=True, right_index=True, validate='one_to_one')

		dataset, _ = train_test_split(tcga_genes_and_2ysurvival, train_size=dataset_size,
			stratify=tcga_genes_and_2ysurvival['X2yr.RF.Surv.'], random_state=random_state)
		return dataset


################### Sample Metabric/TCGA with specific train/valid/test sizes (useful for invetigating the impact of train/valid/test sizes) ###################

def sample_metabric_dataset_vary_sizes(args, train_size, valid_size, test_size):
	"""
	Create dataset splits from Metabric with custom train/valid/test sizes.
	- args: dictionary, or command-line object from argparse
		- args.dataset - dataset_name: name of the dataset (e.g. 'metabric-dr', 'metabric-pam50')
			- 'metabric-dr': dataset with drug response labels
			- 'metabric-pam50': dataset with PAM50 labels
		- args.repeat_id - random seed for the sampling	
	"""
	#### Load expression data
	expressionsMB = pd.read_csv(f'{DATA_DIR}/Metabric_full/MOLECULARDATA/CURTIS_data_Expression.txt', delimiter='\t').T

	# set columns
	expressionsMB.columns = expressionsMB.iloc[0]
	# drop two rows that contain column names
	expressionsMB.drop(expressionsMB.index[[0,1]], inplace=True)
	expressionsMB_genes = expressionsMB.T.copy()

	# load Hallmark gene set
	genes_to_filter = pd.read_csv(f'{DATA_DIR}/imp_genes_list.csv',index_col=0)
	genes_to_filter_unduplicated = genes_to_filter.loc[~genes_to_filter.index.duplicated(keep='first')]

	# keep only the genes from Hallmark
	expressionsMB_filtered = pd.concat([genes_to_filter_unduplicated, expressionsMB_genes],axis=1, join="inner").copy()
	expressionsMB_filtered = expressionsMB_filtered.T.copy().dropna()
	

	#### Load clinical data
	clinMB = pd.read_csv(f'{DATA_DIR}/Metabric_full/MOLECULARDATA/TableS6.txt', delimiter='\t')
	clinMB.set_index('METABRIC.ID',inplace=True)


	#### Set task
	if args.dataset == 'metabric-dr':
		DR = clinMB['DR'].copy().dropna()
		dataset = expressionsMB_filtered.merge(DR, left_index=True, right_index=True, validate='one_to_one')
		label = 'DR'
	elif args.dataset == 'metabric-pam50':
		pam50 = clinMB['Pam50Subtype'].copy().dropna()
		pam50_binary = pam50.map({
			'Basal': int(1), 
			'LumA': int(0),
			'LumB': int(0),
			'Her2': int(0),
			'Normal': int(0)
		}).astype(int)

		dataset = expressionsMB_filtered.merge(pam50_binary, left_index=True, right_index=True, validate='one_to_one')
		label = 'Pam50Subtype'
	else:
		raise ValueError(f'Unknown dataset {args.dataset}')

	return sample_dataset(args, dataset, label, train_size, valid_size, test_size)


def sample_tcga_dataset_vary_sizes(args, train_size, valid_size, test_size):
	"""
	Create dataset splits from TCGA with custom train/valid/test sizes.
	- args: dictionary, or command-line object from argparse
		- args.dataset - dataset_name: name of the dataset (e.g. 'tcga-tumor-grade', 'tcga-2ysurvival')
			- 'tcga-tumor-grade': dataset with tumor grade labels
			- 'tcga-2ysurvival': dataset with 2-year survival labels
		- args.repeat_id - random seed for the sampling	
	"""

	tcga_full = pd.read_csv(f'{DATA_DIR}/TCGA_full/tcga_hncs.csv', index_col=0)
	tcga_full = tcga_full.dropna()

	# filter genes
	partner_genes_to_filter = pd.read_csv(f'{DATA_DIR}/imp_genes_list.csv',index_col=0)
	set_partner_genes_to_filter = set(partner_genes_to_filter.index)

	# Clean the set of columns to match the Partner Naming
	column_names_clean = []
	for column_with_number in tcga_full.columns:
		column_name = column_with_number.split('|')[0]
		column_names_clean.append(column_name)
	
	genes_intersection = list(set(column_names_clean).intersection(set_partner_genes_to_filter))
	genes_intersection = sorted(genes_intersection)

	# keep only the Partner set of genes
	tcga_full_columns_changed = tcga_full.copy()
	tcga_full_columns_changed.columns = column_names_clean
	tcga_only_intersection_genes = tcga_full_columns_changed[genes_intersection]

	if args.dataset == 'tcga-tumor-grade':
		tcga_genes_and_tumor_grade = tcga_only_intersection_genes.copy()
		tcga_genes_and_tumor_grade = tcga_genes_and_tumor_grade.merge(tcga_full['tumor_grade'], left_index=True, right_index=True, validate='one_to_one')
		
		# Keep only G1, G2, G3 classes
		tcga_genes_and_tumor_grade = tcga_genes_and_tumor_grade.loc[tcga_genes_and_tumor_grade['tumor_grade'].isin(['G1', 'G2', 'G3'])]
		tcga_genes_and_tumor_grade['tumor_grade'] = tcga_genes_and_tumor_grade['tumor_grade'].map({'G1': int(0), 'G2': int(1), 'G3': int(2)})
		dataset = tcga_genes_and_tumor_grade
		
		label = 'tumor_grade'
	elif args.dataset == 'tcga-2ysurvival':
		tcga_genes_and_2ysurvival = tcga_only_intersection_genes.copy()
		dataset = tcga_genes_and_2ysurvival.merge(tcga_full['X2yr.RF.Surv.'], left_index=True, right_index=True, validate='one_to_one')

		label = 'X2yr.RF.Surv.'
	else:
		raise ValueError(f'Unknown dataset {args.dataset}')

	return sample_dataset(args, dataset, label, train_size, valid_size, test_size)


def sample_dataset(args, dataset, label, train_size, valid_size, test_size):
	# args: dictionary, or command-line object from argparse

	#### Set train/valid/test sizes
	# Create test set
	dataset_train_valid, dataset_test = train_test_split(dataset, test_size=test_size, 
			random_state=args.repeat_id, shuffle=True, stratify=dataset[label])
	# Create validation set
	dataset_train_large, dataset_valid = train_test_split(dataset_train_valid, test_size=valid_size,
			random_state=args.repeat_id, shuffle=True, stratify=dataset_train_valid[label])
	
	# Create train set (dataset_train contains too many entries. We select only a subset of it)
	if train_size < 1.0:
		dataset_train, _ = train_test_split(dataset_train_large, train_size=train_size,
				random_state=args.repeat_id, shuffle=True, stratify=dataset_train_large[label])
	else:
		dataset_train = dataset_train_large

	return dataset_train[dataset_train.columns[:-1]].to_numpy(), dataset_train[dataset_train.columns[-1]].to_numpy(), \
		   dataset_valid[dataset_valid.columns[:-1]].to_numpy(), dataset_valid[dataset_valid.columns[-1]].to_numpy(), \
		   dataset_test[dataset_test.columns[:-1]].to_numpy(), dataset_test[dataset_test.columns[-1]].to_numpy() 

if __name__ == '__main__':
	# X,y = load_obesity()
	# print(X.shape)
	# print(y.shape)
	# X,y = load_t2d()
	# print(X)
	# # load_t2d()
	# print(X.shape)
	# print(y.shape)
	# print(y)

	# X,y = load('metabric-pam50')
	# print(X.shape)
	# print(y.shape)
	# print(y)

	# tcga1, tcga2 = 'tcga-2ysurvival', 'tcga-tumor-grade'
	# X,y = load(tcga1)
	# print(X.shape)
	# print(y.shape)
	# print(y)
	# # print(X)
	# print(X[:,1])
	# # print(np.isnan(X).any())
	# # print(np.isnan(y).any())

	# # print(np.isnay(X))
	# # print(np.isnan(y))

	# X,y = load("prostate")tcga-2ysurvival
	X,y = load("cll")
	print(y.shape)
	print(y)
	print(np.unique(y))
	print(X)
	print(X.shape)

