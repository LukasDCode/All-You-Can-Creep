import argparse

from .EAsimple import EAsimple
from .executor import Executor

def parse_config():
  parser = argparse.ArgumentParser(description='Run worms with hyper params')
  parser.add_argument('-n','--episodes', type=int, default=2000, help='training episodes')
  parser.add_argument('-v', '--visualize', type=bool, default=False, help='call env.render')
  parser.add_argument('-r', '--result', type=str, default="result", help='file base name to save results into')
  return parser.parse_args()

def main():
  params_eaSimple = {
    "generation_max": 50,
    "mutation_rate": 0.05,
    "crossover_rate": 0.8,
    "population_max": 10,
    "name": "EAsimple"
  }
  config = parse_config()
  domain = WormDomainAdaptor(config)

  with Executor(on_slurm=false, tasks_in_parallel=1, domain) as executor:
    evolution =  EAsimple(executor,domain, params_eaSimple)
    evolution.run()
    
if __name__ == "__main__":
    main()

