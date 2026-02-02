import argparse

class DeclareArg:
  """
  Example use case of this class:
  ```Sample bench_train.py:
  from utils import DeclareArg
  dataset_path = DeclareArg('data_path', str, './data/ETT-small.csv', 'Path to the data directory')
  ```
  Then the bench tool can automatically path the data_path argument to the script and in 
  case you run the script normally, it will use the default value.
  """
  
  def __new__(cls, name, type, default=None, desc=''):
    parse = argparse.ArgumentParser()
    parse.add_argument(f'--{name}', type=type, default=default, help=desc)
    args, _ = parse.parse_known_args()
    return getattr(args, name)