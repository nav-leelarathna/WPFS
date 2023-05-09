import sys
import utils
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import  ModelCheckpoint
import data_module
import copy
from datetime import datetime
from types import SimpleNamespace
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from main import parse_arguments, train

datasets_percentages = [
    ["lung",1],
    ["lung",0.25],
    ["toxicity",1],
    ["toxicity",0.5],
    ["prostate",1],
    ["prostate",0.25],
    ["smk",1],
    ["smk",0.25],
    ["cll",1],
    ["cll",0.5],
    ["lung_gordon",1],
    ["lung_gordon",0.25],
    ["breast",1],
    ["breast",0.5],
    ["metabric-pam50",1],
    ["metabric-pam50",0.05],
]

def dietnetworks(sweep_id=None, name='dietnetworks'):
    project="baselines"
    base_configuration = utils.parse("configs/config.json")
    if sweep_id is None:
        sweep_configuration = {
            'method': 'grid',
            'name': name,
            'metric': {
                'goal': 'minimize', 
                'name': 'val_loss'
                },
            'parameters': {
                "dataset_percentages" : {"values" :[ datasets_percentages[0]]},
                "seed_kfold" : {"values": [0]},
                "split_id" : {"values": [0]},
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
    else:
        sweep_id = f"nav-leelarathna/{project}/{sweep_id}"
    wandb.agent(sweep_id, function=lambda : _model(project, base_configuration, name="dietnetworks"))

def fsnet(sweep_id=None, name='fsnet'):
    project="baselines"
    base_configuration = utils.parse("configs/config.json")
    if sweep_id is None:
        sweep_configuration = {
            'method': 'grid',
            'name': name,
            'metric': {
                'goal': 'minimize', 
                'name': 'val_loss'
                },
            'parameters': {
                "dataset_percentages" : {"values" :[ datasets_percentages[0]]},
                "seed_kfold" : {"values": [0]},
                "split_id" : {"values": [0]},
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
    else:
        sweep_id = f"nav-leelarathna/{project}/{sweep_id}"
    wandb.agent(sweep_id, function=lambda : _model(project, base_configuration, name="fsnet"))

def _model(project, config, name):
    wandb_logger = WandbLogger(project=project,log_model=True, save_dir="runs")
    split_id = wandb_logger.experiment.config.split_id
    seed_kfold = wandb_logger.experiment.config.seed_kfold
    dataset_percentage = wandb_logger.experiment.config.dataset_percentages[1]
    datasetName = wandb_logger.experiment.config.dataset_percentages[0]
    run_name = f"{name}_{datasetName}_{dataset_percentage}"
    wandb_logger.experiment.name = run_name
    base_configuration = copy.deepcopy(config)
    base_configuration.data_module.seed_kfold = seed_kfold
    base_configuration.data_module.split_id = split_id
    base_configuration.data_module.dataset_percentage = dataset_percentage
    base_configuration.data_module.name= datasetName
    dict_config = utils.namespace_to_dict(base_configuration)
    wandb_logger.experiment.config.update(dict_config)
    # utils.write_config(base_configuration, project, run_name)
    args = parse_arguments()
    args.dataset = datasetName
    args.model = name
    dm = data_module.create_datamodule(base_configuration, args)
    train(args, wandb_logger, dm)
    # model = utils.init_obj(base_configuration.model, init_type='Model')
    # model.configure_optimizers(utils.parse_class(base_configuration.train.optimizer), base_configuration.train.learning_rate)
    # wandb_logger.watch(model)
    # train(wandb_logger, model, dataset, base_configuration, patience=150)

def classifier(latent_sweep_id=None, classifier_sweep_id=None):
    project="classifier"
    if classifier_sweep_id is None:
        sweep_runs,original_sweep_name = utils.get_sweep_runs("baselines", latent_sweep_id)
        name =  "classifier_"+original_sweep_name        
        sweep_config = {
            'method': 'grid',
            'name': name,
            'metric': {
                'goal': 'minimize', 
                'name': 'val_loss'
                },
            'parameters': {
                "seed_validation" : {"values": [0,1,2,3,4]},
                "sweep_runs": {"values" : sweep_runs}
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project=project)
    else:
        sweep_id = f"nav-leelarathna/{project}/{classifier_sweep_id}"
    classifier_config = utils.parse("configs/classifier_config.json")
    wandb.agent(sweep_id, function=lambda : _classifier(project, classifier_config))

def _classifier(project, classifier_config):
    wandb_logger = WandbLogger(project=project,log_model=True, save_dir="runs")
    sr = wandb_logger.experiment.config.sweep_runs
    seed_validation = wandb_logger.experiment.config.seed_validation

    original_sweep_config = utils.getRunConfig(sr[0], sr[3]) #= sr[3]
    # print(original_sweep_config)
    # print(original_sweep_config)
    data_module = original_sweep_config.data_module 
    # print(data_module)
    classifier_config.data_module = data_module 
    classifier_config.data_module.latent.enabled = True 
    if hasattr(classifier_config.data_module.latent.args, "project_name"):
        delattr(classifier_config.data_module.latent.args, "project_name")
    classifier_config.data_module.latent.args.project = sr[0]
    # classifier_config.data_module.latent.args.run_name = sr[1]
    classifier_config.data_module.latent.args.model_name = sr[2]
    # classifier_config.data_module.latent.args.configuration = sr[3]
    classifier_config.data_module.latent.args.run_id = sr[3]
    if hasattr(classifier_config.data_module.latent.args, "run_name"):
        delattr(classifier_config.data_module.latent.args, "run_name")
    classifier_config.data_module.seed_validation = seed_validation
    run_name = "classifier_"+sr[1]
    wandb_logger.experiment.name = run_name
    train(classifier_config, "classifier", run_name)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        globals()[sys.argv[1].split("=")[-1]]()
    elif len(sys.argv) == 3:
        globals()[sys.argv[1].split("=")[-1]](str(sys.argv[2].split("=")[-1]))
    elif len(sys.argv) == 4:
        globals()[sys.argv[1].split("=")[-1]](str(sys.argv[2].split("=")[-1]),str(sys.argv[3].split("=")[-1]))
    else:   
        print("unrecognised number of arguments")