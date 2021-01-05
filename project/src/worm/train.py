import argparse

from .domain import WormDomainAdaptor
from ..exec.executor import Executor


def create_parser():
  parser = argparse.ArgumentParser(description='Run worms with hyper params')
  parser.add_argument('-a','--alpha', type=float, default=0.001, help='the learning rate')
  parser.add_argument('-g','--gamma', type=float, default=0.999 , help='the discount factor for rewards')
  parser.add_argument('-e', '--entropy', type=float, default=1e-4, help='the exploitation rate')
  parser.add_argument("-sd", '--state_dict', type=str, default=None, help='the existing state dict to load')
  parser.add_argument("-c","--continue_training", default=False, action='store_true', help='whether to continue training from state dict')

  WormDomainAdaptor.add_parse_args(parser)
  Executor.add_parser_args(parser)
  return parser


def main():
  parser = create_parser()
  config = parser.parse_args()
  if config.continue_training and not config.state_dict:
    parser.error("--continue_training requires --state_dict")

  print("Run with {}", str(config))
  domain = WormDomainAdaptor(config)

  with Executor(config=config, domain=domain) as executor:
    # Hyperparameters
    params = {}
    params["gamma"] = config.gamma
    params["alpha"] = config.alpha
    params["entropy"] = config.entropy
    future = executor.submit_task(
      params=params,
      state_dict=config.state_dict,
      continue_training=config.continue_training,
    )
    print(future.get())

if __name__ == "__main__":
  main()