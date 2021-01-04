import argparse
from ..worm.domain import WormDomainAdaptor
from .EAsimple import EAsimple
from .Gridsearch import Gridsearch
from ..exec.executor import Executor

def parse_config():
    parser = argparse.ArgumentParser(description='Run worms with hyper params')
    Executor.add_parser_args(parser)
    WormDomainAdaptor.add_parse_args(parser)
    parser.add_argument('-variant', '--variant', required=False, type=str, choices=["evolution", "gridsearch"], default="evolution",
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
    domain = WormDomainAdaptor(config)

    with Executor(config=config, domain=domain) as executor:
        if config.variant == "evolution":
            evolution = EAsimple(executor, domain, params_eaSimple, result_dir=config.result_dir)
            evolution.run()
        else:
            gridsearch = Gridsearch(executor, domain,)
            gridsearch.run()

if __name__ == "__main__":
    main()
