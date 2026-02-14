import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

import train_GAN
import cfg
from train_GAN import main_worker

from bench_utils import DeclareArg
import bench

dataset_path = DeclareArg('data_path', str, './data/ETTh1.csv', 'Path to the data file')

OUTPUT_DIR = DeclareArg('output_dir', str, './_bench_output', 'Output directory for results')
EXPERIMENT_NAME = DeclareArg('experiment_name', str, 'experiment', 'Experiment name')
RUN_NAME = DeclareArg('run_name', str, 'run_0', 'Run name')
SEED = DeclareArg('seed', int, 0, 'Random seed')

# Original project is hard-coded with unimib_load_dataset to be the name of the instance of the dataset class.
# A regular torch Dataset class needs to be created and this value needs to be overridden with instance of the new Dataset class.
class SimpleETTh1(Dataset):
  def __init__(self, *args, **kwargs):
    # We load the different parts of CSV dataset based on data_mode argument 
    data_mode = kwargs.get('data_mode', 'Train')
    
    # TTS-GAN expects 3 time series columns
    # We will use HUFL, MUFL, LUFL columns from ETTh1 dataset for now (hard-coded)
    df = pd.read_csv(dataset_path)
    values = df[['HUFL', 'MUFL', 'LUFL']].values.astype(np.float32)
    
    mean, std = values.mean(axis=0), values.std(axis=0)
    values = (values - mean) / (std + 1e-8)
    
    n = len(values)
    if data_mode == 'Train':
      values = values[:int(n*0.7)]
    else:
      values = values[int(n*0.8):]
    
    seq_len = 150
    windows = []
    for i in range(len(values) - seq_len + 1):
      window = values[i:i+seq_len].T.reshape(3, 1, seq_len)
      windows.append(window)
    
    self.data = np.array(windows, dtype=np.float32)
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return torch.from_numpy(self.data[idx]), torch.tensor(0, dtype=torch.long)

def main():
  
  args = cfg.parse_args()
  
  args.seed = True
  args.random_seed = SEED
  # args.gpu = torch.device('cuda:0') if torch.cuda.is_available() else None

  # If args.rank == 0 then args.path_helper is populated else 
  # args.path_helper must be passed
  args.rank = 0

  args.distributed = False
  args.class_name = 'ETTh1'
  
  args.max_epoch = DeclareArg('max_epoch', int, 1, 'Number of epochs to train')
  args.max_iter = DeclareArg('max_iter', int, 1000, 'Number of iterations to train')
  args.gen_batch_size = DeclareArg('gen_batch_size', int, 16, 'Generator batch size')
  args.dis_batch_size = DeclareArg('dis_batch_size', int, 16, 'Discriminator batch size')
  args.batch_size = DeclareArg('batch_size', int, 16, 'Overall batch size')
  args.num_workers = 0
  args.gen_model = 'my_gen'
  args.dis_model = 'my_dis'
  args.latent_dim = DeclareArg('latent_dim', int, 100, 'Dimensionality of the latent space')
  args.g_lr = DeclareArg('g_lr', float, 0.0003, 'Learning rate for the generator')
  args.d_lr = DeclareArg('d_lr', float, 0.0003, 'Learning rate for the discriminator')
  args.optimizer = DeclareArg('optimizer', str, 'adam', 'Optimizer to use (adam/sgd)')
  args.wd = DeclareArg('wd', float, 0.0001, 'Weight decay (L2 regularization)')
  args.beta1 = DeclareArg('beta1', float, 0.9, 'Beta1 for Adam optimizer')
  args.beta2 = DeclareArg('beta2', float, 0.999, 'Beta2 for Adam optimizer')
  args.n_critic = 1
  args.val_freq = 100000
  args.show = False # If set to True, bug occurs since `fid_stat` is undefined in the original train_GAN.py
  args.init_type = DeclareArg('init_type', str, 'xavier', 'Weight initialization type')
  args.exp_name = 'quick_test_etth1'
  args.load_path = None

  args.grow_steps = [0, 0]

  bench.start_run()

  # Overriding the hard-coded data loading variable with the proper Dataset class.
  train_GAN.unimib_load_dataset = SimpleETTh1

  # Run training
  ngpus_per_node = torch.cuda.device_count()
  main_worker(args.gpu, ngpus_per_node, args)
  
  bench.end_run()

if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    print(f"Error: {e}")
    bench.end_run("FAILED")

