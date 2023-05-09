import json
from collections import OrderedDict
import os
from datetime import datetime
from pathlib import Path
from functools import partial
import importlib
from types  import FunctionType
import wandb
from pprint import pprint
from types import SimpleNamespace
from pprint import pprint
from argparse import Namespace
import torch
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
from models import GeneralNeuralNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def flatten(l):
    return [item for sublist in l for item in sublist]
    
def generate_overlapping_sets(num_features=15, subsets=2, percentage_overlap=0.5):
    lengthSubsets = num_features / (subsets - ((subsets-1)*percentage_overlap))
    # print(lengthSubsets)
    indices = []
    start = 0
    for _ in range(subsets):
        indices.append([math.ceil(start), math.floor(start+lengthSubsets)])
        start += (1-percentage_overlap)*lengthSubsets
    assert indices[0][0] == 0
    assert indices[-1][-1] == num_features
    return indices

def test_overlapping_set():
    data = np.arange(start=100,stop=160)
    data.shape = (4,15)
    num_features = data.shape[1]
    print(data)
    start_stop_indices = generate_overlapping_sets(num_features,2,0.5)
    indices =np.arange(num_features)
    np.random.shuffle(indices)
    subset_indices = [indices[start_stop[0]:start_stop[1]] for start_stop in start_stop_indices]
    # TODO save the above indices for later use
    arrays = [data[:,s_i] for s_i in subset_indices]
    print(arrays)
    print([arr[2,:] for arr in arrays])


def num_params_vae(N, l, k):
  return 2 * (N * l + l * l + l * k) + l * k


def compute_l_vae(N, k, params):
  return int((-(2 * N + 3 * k) / 4) + math.sqrt(params / 2 +
                                                ((2 * N + 3 * k) / 4)**2))

def computeEquivalentMvaeLayerWidth(F, width, k, F_i):
    vae_params = F*width + width**2 + width*k 
    mvae_params = F_i * vae_params / F
    middle_term = (F_i + k) / 2 
    a = math.sqrt(mvae_params + middle_term**2) - middle_term
    return int(a)

def computeEquivalentFsmvaeLayerWidth(F, width, z, learners):
    vae_params = F*width + width**2 + width*z 
    fsmvae_params = vae_params / learners
    middle_term = (F + z) / 2 
    a = math.sqrt(fsmvae_params + middle_term**2) - middle_term
    return int(a)

def load_config_from_run(project, run_name):
    config_path = f"experiments/{project}/{run_name}/config.json"
    configuration = parse(config_path=config_path)
    return configuration

def load_model_from_wandb(project, run_id, model_name):
    # if configuration is None:
    #     configuration = load_config_from_run(project, run_name)
    configuration = getRunConfig(project,run_id)
    model_name = project + "/" + model_name
    api = wandb.Api()
    artifact = api.artifact(model_name)
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, "model.ckpt")
    # model_class = parse_class(configuration.model.name)
    model = GeneralNeuralNetwork.load_from_checkpoint(model_path)
    return model

def getRunConfig(project,run_id):
    overrides = {
        "entity" : "nav-leelarathna",
    }
    api = wandb.Api(overrides=overrides)
    run = api.run(project + '/' + run_id)
    configuration = dict_to_namespace(run.config)
    return configuration

def get_sweep_runs(project, sweep_id):
    overrides = {
        "entity" : "nav-leelarathna",
        "project" : project
    }
    api = wandb.Api(overrides=overrides)
    sweep = api.sweep(f"{project}/{sweep_id}")
    sweep_name = sweep.config["name"]
    runs = sweep.runs
    artifact_dicts =[]
    for run in runs:
        artifact_name = f"model-{run.id}:v0"
        # config = dict_to_namespace(run.config)
        information = [project, run.name, artifact_name, run.id]      
        artifact_dicts.append(information)
    return artifact_dicts, sweep_name

def write_config(configuration, project, run_name):
    mkdirs(f"experiments/{project}/{run_name}")
    path = f"experiments/{project}/{run_name}/config.json"
    config = namespace_to_dict(configuration)
    # pprint(config)
    fname = Path(path)
    with fname.open('wt') as handle:
        json.dump(config, handle, indent=4, sort_keys=False)

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def namespace_to_dict(namespace):
    my_dict = {}
    namespace = namespace.__dict__
    for (key, value) in namespace.items():
        if isinstance(value, SimpleNamespace):
            value = namespace_to_dict(value)
        elif isinstance(value, list):
            value = [namespace_to_dict(f) if isinstance(f, SimpleNamespace) else f for f in value]
            # print(key,value)
        my_dict[key] = value
    return my_dict

def dict_to_namespace(d):
    opt = json.loads(json.dumps(d), object_hook=lambda d: SimpleNamespace(**d))
    return opt

def parse(config_path):
    json_str = ''
    with open(config_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    namespace_opt = json.loads(json_str, object_hook=lambda d: SimpleNamespace(**d))
    return namespace_opt


def parse_class(name):
    file_name, class_name = name[0], name[1]
    module = importlib.import_module(file_name)
    ret =  getattr(module, class_name)
    ret.__name__  = ret.__class__.__name__
    return ret

def init_obj(opt, *args, default_file_name='default file', given_module=None, init_type='Network', name="name", **modify_kwargs):
    # if opt is None or len(opt)<1:
    #     # logger.info('Option is None when initialize {}'.format(init_type))
    #     return None
    
    ''' default format is dict with name key '''
    # if isinstance(opt, str):
    #     opt = {'name': opt}
        # logger.warning('Config is a str, converts to a dict {}'.format(opt))

    # name = opt[name]
    # if name is None:
    #     name = opt.name
    # else:
    name = getattr(opt, name)
    ''' name can be list, indicates the file and class name of function '''
    if isinstance(name, list):
        file_name, class_name = name[0], name[1]
    else:
        file_name, class_name = default_file_name, name
    try:
        if given_module is not None:
            module = given_module
        else:
            module = importlib.import_module(file_name)
        
        attr = getattr(module, class_name)
        # kwargs = opt.get('args', {})
        kwargs = {}
        if hasattr(opt, "args"):
            kwargs = opt.args.__dict__

        kwargs.update(modify_kwargs)
        ''' import class or function with args '''
        if isinstance(attr, type): 
            ret = attr(*args, **kwargs)
            ret.__name__  = ret.__class__.__name__
        elif isinstance(attr, FunctionType): 
            ret = partial(attr, *args, **kwargs)
            ret.__name__  = attr.__name__
            # ret = attr
        # logger.info('{} [{:s}() from {:s}] is created.'.format(init_type, class_name, file_name))
    except Exception as e:
        print(type(e))
        raise NotImplementedError('{} [{:s}() from {:s}] not recognized. Exception thrown {}'.format(init_type, class_name, file_name, e))
    print(f"[-] Initialised object of type [{type(ret)}]. Initialisation category: [{init_type}]")
    return ret

def initialise_networks(networks, input_sizes=None):
    encoder_networks = []
    decoder_networks = []
    for i, net in enumerate(networks):
        if input_sizes is None:
            # encoder = init_obj(net, logger, name="encoder")
            encoder = init_obj(net, name="encoder")
            decoder = init_obj(net, name="decoder")
            # decoder = init_obj(net, logger, name="decoder")
        else:
            # encoder = init_obj(net, logger, input_size=input_sizes[i], name="encoder")
            encoder = init_obj(net, input_size=input_sizes[i], name="encoder")
            decoder = init_obj(net, input_size=input_sizes[i],name="decoder")
            # decoder = init_obj(net, logger, input_size=input_sizes[i],name="decoder")
        encoder_networks.append(encoder)
        decoder_networks.append(decoder)
    return encoder_networks, decoder_networks

def is_array_in_list(arr, arr_list):
    """Checks if a trial array is in a list of arrays."""
    for element in arr_list:
        if np.array_equal(element, arr):
            return True
    return False

def reparameterize(training, mu, logvar):
    """Reparameterization for multivariate Gaussian posteriors.

    Args:
        training: bool, indicating if training or testing.
        mu: location parameters.
        logvar: scale parameters (log of variances).

    Returns:
        Reparameterized representations.
    """
    if training:
        std = logvar.mul(0.5).exp_()
        eps = (
            torch.randn(std.size()).to(mu)
        )
        return eps.mul(std).add_(mu)
    else:
        return mu

def compute_tc(tc_tuple, style_mu, style_logvar, content_mu, content_logvar, train=True, dimperm=False):
    """
    Estimates total correlation (TC) between a set of variables and optimizes
    the TCDiscriminator if train=true.
    NOTE: adapted from FactorVAE (https://github.com/1Konny/FactorVAE)

    Args:
        tc_tuple: tuple containing a TCDiscriminator and its optimizer
        style_mu: location parameter of modality-specific Gaussian posterior
        style_logvar: scale parameter (log variance) of modality-specific Gaussian posterior
        content_mu: location parameter of shared Gaussian posterior
        content_logvar: scale parameter (log variance) of shared Gaussian posterior
        train: boolean indicator if training or testing
        dimperm: whether to permute the individual dimensions of the
            modality-specific representation. Default: False.

    Returns:
        A tuple (tc, d_loss), where tc is the estimated total correlation and
        d_loss is the loss of the cross-entropy loss of the discriminator
    """
    # prep
    tc_d, tc_opt = tc_tuple
    num_samples = style_mu.shape[0]
    zeros = torch.zeros(num_samples, dtype=torch.long).to(device)
    ones = torch.ones(num_samples, dtype=torch.long).to(device)
    tc_opt.zero_grad()
    if train is True:
        tc_d.train()
    else:
        tc_d.eval()
    # print(f"mu grad is {content_mu.grad_fn}")

    # reparameterize to get representations
    s = reparameterize(training=train, mu=style_mu, logvar=style_logvar)
    c = reparameterize(training=train, mu=content_mu, logvar=content_logvar)

    # print(f"s grad is {s.grad_fn}")
    # permute the second representation
    s_perm = s.clone()
    if dimperm:
        for i in range(s_perm.shape[-1]):
            s_perm[:, i] = s_perm[torch.randperm(num_samples), i]
    else:  # batch-wise permutation, keeping dimensions intact
        s_perm = s_perm[torch.randperm(num_samples)]

    # compute the CEL and backprop within the discriminator

    scores = tc_d(s.data, c.data)
    scores_perm = tc_d(s_perm.data, c.data)
    # print(f"scores grad is {scores.grad_fn}")

    d_loss = 0.5 * (F.cross_entropy(scores, zeros) + F.cross_entropy(scores_perm, ones))
    d_loss = Variable(d_loss, requires_grad = True)
    # backprop
    # print(f"loss grad is { d_loss.grad_fn}")
    if train is True:
        d_loss.backward()
        tc_opt.step()
        # pass

    # estimate tc
    print(s.shape)
    print(c.shape)
    scores = tc_d(s, c)
    lsm = F.log_softmax(scores, dim=1)
    tc = (lsm[:, 0] - lsm[:, 1]).mean()
    # sm = F.softmax(scores,dim=1)
    print(torch.min(lsm))
    print(torch.max(lsm))
    # tc = (sm[:,0] - sm[:,1]).mean()
    print(tc)
    return tc, d_loss


def compute_infomax(projection_head, h1, h2, tau=1.0):
    """
    Estimates the mutual information between a set of variables.
    Automatically uses $K = batch_size - 1$ negative samples.

    Args:
        projection_head: projection head for the MI-estimator. Can be identity.
        h1: torch.Tensor, first representation
        h2: torch.Tensor, second representation
        tau: temperature hyperparameter.

    Returns:
        A tuple (mi, d_loss) where mi is the estimated mutual information and
        d_loss is the cross-entropy loss computed from contrasting
        true vs. permuted pairs.
    """

    # compute cosine similarity matrix C of size 2N * (2N - 1), w/o diagonal elements
    batch_size = h1.shape[0]
    z1 = projection_head(h1)
    z2 = projection_head(h2)
    z1_normalized = F.normalize(z1, dim=-1)
    z2_normalized = F.normalize(z2, dim=-1)
    z = torch.cat([z1_normalized, z2_normalized], dim=0)  # 2N * D
    C = torch.mm(z, z.t().contiguous())  # 2N * 2N
    # remove diagonal elements from C
    mask = ~ torch.eye(2 * batch_size, device=C.device).type(torch.ByteTensor)  # logical_not on identity matrix
    C = C[mask].view(2 * batch_size, -1)  # 2N * (2N - 1)

    # compute loss
    numerator = 2 * torch.sum(z1_normalized * z2_normalized) / tau
    denominator = torch.logsumexp(C / tau, dim=-1).sum()
    loss = (denominator - numerator) / (2 * batch_size)
    # Use the second term for the loss 
    return np.nan, loss  # NOTE: Currently returns MI=NaN

def visualiseGroups(groupFile= "groups_epoch_20.csv"):
    from matplotlib import pyplot as plt
    import csv
    with open(groupFile, "r") as f:
        reader = csv.reader(f, delimiter=',')
        groups = [[int(i) for i in row] for row in reader]
        print(groups)

    colours = [[255,0,0],[0,255,0],[0,0,255],[255,255,0]]
    data = np.zeros( (28,28,3), dtype=np.uint8)
    for i, group in enumerate(groups):
        for index in group:
            row = math.floor(index/28)
            col = index % 28 
            data[row,col] = colours[i]
    # data[27,27] = [255,0,0]
    plt.imshow(data, interpolation='nearest')
    plt.show() 

def visualiseGroups2(groupFile= "csvFiles/groups_epoch_178.csv"):
    from matplotlib import pyplot as plt
    import csv
    with open(groupFile, "r") as f:
        reader = csv.reader(f, delimiter=',')
        groups = [[float(i) for i in row] for row in reader]
        # print(groups)

    colours = [[255,0,0],[0,255,0],[0,0,255],[255,255,0]]
    data = np.zeros( (28,28,3), dtype=np.uint8)
    for j in range(28*28):
        group_probabilities = [g[j] for g in groups]
        probSum = min(group_probabilities)
        group_probabilities = [g/probSum for g in group_probabilities]
        if j < 28:
            print(group_probabilities)
        maxIndex = 0 
        for k in range(1, len(group_probabilities)):
            if group_probabilities[k] > group_probabilities[maxIndex]:
                maxIndex = k 
        row = math.floor(j/28)
        col = j % 28 
        data[row,col] = colours[maxIndex]
    # for i, group in enumerate(groups):
    #     for index in group:
    # # data[27,27] = [255,0,0]
    plt.imshow(data, interpolation='nearest')
    plt.show() 

def totalCorrelation(latentDataset):
    num, dim = latentDataset.shape 
    print(f"num: {num}, latent dim: {dim}")
    covariance = np.cov(latentDataset.T)
    print(covariance.shape)
    return 0.5 * (np.sum(np.log(np.diag(covariance))) - np.linalg.slogdet(covariance)[1])




if __name__ == "__main__":
    # config = load_config_from_run("mvae_test", "run_2023_01_23_22_51_59")
    # configuration = parse("configs/mvae_config2.json")
    # import numpy as np
    # import data_module
    # dataset = data_module.create_datamodule(configuration)
    # model = init_obj(configuration.model, init_type='Model')
    # d = namespace_to_dict(configuration)
    # pprint(d)
    # write_config(configuration, "test.json")
    visualiseGroups2("csvFiles/groups_epoch_180.csv")
    # latentDataset = np.random.normal(2  ,1,size=(123,64))
    # tc = totalCorrelation(latentDataset)
    # print(tc)