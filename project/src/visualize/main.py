import argparse
import json
import math
import matplotlib.pyplot as plot

def visualize_simple(config, data_key, result_set):
    columns = config.columns
    rows = math.ceil(len(result_set)/columns)

    fig, axs = plot.subplots(columns, rows, figsize=(16,24), sharey=True, sharex=True)
    for ax in axs.flat():
        ax.set(xLabel='episode', ylabel="return")
        ax.label_outer()
    fig.supertitle("{}-{}".format(config.name,data_key))
    
    for index, result in enumerate(result_set):
        subfig = axs[
            index % rows,
            math.floor((index - index % rows)/rows),
        ]
        subfig.set_title("{} {}".format(
            result["algorithm"], str(result["params"])
        ))
        data = result[data_key]
        subfig.plot(range(len(data)), data)
    
    if config.file:
        plot.savefig(config.file)
    else:
        fig.show()

def parse_config():
    parser = argparse.ArgumentParser("Visualize worms results")
    parser.add_argument(
        '-n', '--name',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-f', '--file', 
        type=argparse.FileType("w"),
        default=None,
        )
    parser.add_argument(
        '-c', '--columns', 
        type=int,
        default=3,
        help="number of columns to display",
        )
    parser.add_argument(
        '-r', '--results',
        type=argparse.FileType("r"),
        nargs='+',
        )
    reward_analysis = parser.add_subparsers("rewards", dest="sub", required=True)
    return parser.parse_args()
    

def main():
    config = parse_config()
    result_sets = [json.load(file) for file in config.results]
    switch = {
        "rewards": lambda: visualize_simple(config, "rewards", result_sets), 
    }
    # execute subcommand 
    switch.get(config.sub)()

if __name__ == "__main__":
    main()
