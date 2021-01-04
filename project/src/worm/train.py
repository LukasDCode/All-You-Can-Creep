import argparse

from .domain import WormDomainAdaptor
from ..exec.executor import Executor


def parse_config():
  parser = argparse.ArgumentParser(description='Run worms with hyper params')
  parser.add_argument('-a','--alpha', type=float, default=0.001, help='the learning rate')
  parser.add_argument('-g','--gamma', type=float, default=0.999 , help='the discount factor for rewards')
  parser.add_argument('-e', '--entropy', type=float, default=1e-4, help='the exploitation rate')
  parser.add_argument("-asd", '--agent_state_dict', type=str, default=None, help='the existing agent state dict to load')

  WormDomainAdaptor.add_parse_args(parser)
  Executor.add_parser_args(parser)
  return parser.parse_args()


def main():
  config = parse_config()
  print("Run with {}", str(config))
  domain = WormDomainAdaptor(config)

  with Executor(config=config, domain=domain) as executor:
    # Hyperparameters
    params = {}
    params["gamma"] = config.gamma
    params["alpha"] = config.alpha
    params["entropy"] = config.entropy
    future = executor.submit_task(params=params)
    print(future.get())

if __name__ == "__main__":
  main()