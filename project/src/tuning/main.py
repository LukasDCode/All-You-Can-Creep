import argparse
from ..worm.training import AgentRunner
from ..worm.domain import WormDomain
from .EAsimple import EAsimple
from .Gridsearch import Gridsearch
from ..exec.executor import Executor
from ..agents.a2c import A2CLearner 
from ..agents.ppo import PPOLearner
from ..agents.ppo_v2 import PPOv2Learner
import mlflow

def create_parser():
    parser = argparse.ArgumentParser(description='Run worms with hyper params')
    subparser = parser.add_subparsers(title="agents", description="Which agent to run", dest="agent", required=True)

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

    parser.add_argument('-variant', '--variant', required=False, type=str, choices=["evolution", "gridsearch"], default="gridsearch",
                        help="choice of tuningvariant")
    return parser

def main():
    params_eaSimple = {
        "generation_max": 50,
        "mutation_rate": 0.05,
        "crossover_rate": 0.8,
        "population_max": 10,
        "name": "EAsimple"
    }
    parser =  create_parser() 
    config = parser.parse_args() 
    kwargs = vars(config)

    print("Run with {}", str(config))
    if config.agent == "a2c":
      agent_class=A2CLearner
    elif config.agent == "ppo":
      agent_class=PPOLearner
    elif config.agent == "ppo_v2":
      agent_class=PPOv2Learner
    else:
      parser.error("no valid agent")
      return



    runner = AgentRunner(**kwargs)
    mlflow.set_experiment(config.result_dir)
    with Executor(runner=runner, **kwargs) as executor:
        if config.variant == "evolution":
            evolution = EAsimple(
                executor=executor,
                agent_class=agent_class,
                param_dictionary=params_eaSimple,
                result_dir=config.result_dir
            )
            evolution.run()
        else:
            gridsearch = Gridsearch(
                executor=executor,
                agent_class=agent_class,
                **kwargs,
            )
            gridsearch.run()

if __name__ == "__main__":
    main()
