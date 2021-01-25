import argparse
from ..worm.training import AgentRunner
from ..worm.domain import WormDomain
from .EAsimple import EAsimple
from .Gridsearch import Gridsearch
from ..exec.executor import Executor
from ..agents.a2c import A2CLearner
import mlflow

def parse_config():
    parser = argparse.ArgumentParser(description='Run worms with hyper params')
    WormDomain.add_parse_args(parser)
    Executor.add_parser_args(parser)
    AgentRunner.add_parse_args(parser)
    A2CLearner.add_config_args(parser)
    A2CLearner.add_hyper_param_args(parser)
    parser.add_argument('-variant', '--variant', required=False, type=str, choices=["evolution", "gridsearch"], default="gridsearch",
                        help="choice of tuningvariant")
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
    kwargs = vars(config)


    runner = AgentRunner(**kwargs)
    mlflow.set_experiment(config.result_dir)
    with Executor(runner=runner, **kwargs) as executor:
        if config.variant == "evolution":
            evolution = EAsimple(
                executor=executor,
                agent_class=A2CLearner,
                param_dictionary=params_eaSimple,
                result_dir=config.result_dir
            )
            evolution.run()
        else:
            gridsearch = Gridsearch(
                executor=executor,
                agent_class=A2CLearner,
                **kwargs,
            )
            gridsearch.run()

if __name__ == "__main__":
    main()

