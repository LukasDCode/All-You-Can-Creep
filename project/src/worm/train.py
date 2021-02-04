import argparse
from .domain import WormDomain
from matplotlib.pyplot import title
from ..exec.executor import Executor
from ..agents.a2c import A2CLearner
from ..agents.randomagent import RandomAgent
from ..agents.ppo import PPOLearner
from ..agents.ppo_v2 import PPOv2Learner
from .training import AgentRunner

def create_parser():
  parser = argparse.ArgumentParser(description='Train agent')
  # add agents subcommands
  subparser = parser.add_subparsers(title="agents", description="Which agent to run", dest="agent", required=True)

  random = subparser.add_parser("rand")
  AgentRunner.add_parse_args(random)
  Executor.add_parser_args(random)
  WormDomain.add_parse_args(random)
  
  a2c_parser = subparser.add_parser("a2c")
  AgentRunner.add_parse_args(a2c_parser)
  Executor.add_parser_args(a2c_parser)
  A2CLearner.add_hyper_param_args(a2c_parser)
  A2CLearner.add_config_args(a2c_parser)
  WormDomain.add_parse_args(a2c_parser)

  ppo_parser = subparser.add_parser("ppo")
  AgentRunner.add_parse_args(ppo_parser)
  Executor.add_parser_args(ppo_parser)
  PPOLearner.add_hyper_param_args(ppo_parser)
  PPOLearner.add_config_args(ppo_parser)
  WormDomain.add_parse_args(ppo_parser)

  ppo_v2_parser = subparser.add_parser("ppo_v2")
  AgentRunner.add_parse_args(ppo_v2_parser)
  Executor.add_parser_args(ppo_v2_parser)
  PPOv2Learner.add_hyper_param_args(ppo_v2_parser)
  PPOv2Learner.add_config_args(ppo_v2_parser)
  WormDomain.add_parse_args(ppo_v2_parser)

  return parser

def main():
  parser = create_parser()
  config = parser.parse_args()
  if config.continue_training and not config.state_dict:
    parser.error("--continue_training requires --state_dict")

  print("Run with {}", str(config))
  if config.agent == "a2c":
    agent_class=A2CLearner
  elif config.agent == "rand":
    agent_class=RandomAgent
  elif config.agent == "ppo":
    agent_class=PPOLearner
  elif config.agent == "ppo_v2":
    agent_class=PPOv2Learner
  else:
    parser.error("no valid agent")
    return

  kwargs = vars(config)
  runner = AgentRunner(**kwargs)
  with Executor(runner=runner,**kwargs) as executor:
    future = executor.submit_task(agent_class=agent_class, **kwargs,)
    future.get()

if __name__ == "__main__":
  main()