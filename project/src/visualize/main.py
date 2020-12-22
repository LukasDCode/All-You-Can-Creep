import argparse
import json
import math
import matplotlib.pyplot as plot


def visualize_single(config, ):
    result_set = [json.load(file) for file in config.result_file]
    data_key = config.measure
    print("visualizing ", data_key)

    columns = config.columns
    rows = math.ceil(len(result_set)/columns)

    fig, axs = plot.subplots(
        nrows=rows, 
        ncols=columns, 
        squeeze=False, 
        figsize=(columns*4,rows*4), 
        sharex=True,
        sharey=True,)

    for ax in axs.flat:
        ax.set(xLabel='episode', yLabel="return")
        ax.label_outer()
    fig.suptitle("{}-{}".format(config.name,data_key))
    
    for index, result in enumerate(result_set):
        column, row = index % columns,math.floor((index - index % columns)/columns),
        subfig = axs[row, column]
        subfig.set_title("{} {}".format(
            result["algorithm"], str(result["params"])
        ))
        data = result["measures"][data_key]
        subfig.plot(range(len(data)), data)
    
    if config.file:
        plot.savefig(config.file)
    else:
        plot.show()

def visualize_all(config):
    result_sets = [json.load(file) for file in config.result_file]

    measures = []
    for set in result_sets:
        measures.extend([m for m,_ in set["measures"].items() if m not in measures])
    print("visualizing measures ", measures,)


    columns = len(result_sets)  
    rows = max([len(x["measures"]) for x in result_sets]) 
    fig, axs = plot.subplots(
        nrows=rows, 
        ncols=columns, 
        squeeze=False, 
        figsize=(columns*4,rows*4), 
        sharex=True,
        sharey='row',
    )
    fig.suptitle(config.name)
    for column, data_set in enumerate(result_sets):
        for measure_name, measure_data in data_set["measures"].items():
            row = measures.index(measure_name)
            subfig = axs[row, column]
            subfig.set_title("{} {}".format(str(data_set["algorithm"]),str(data_set["params"])) )
            subfig.set(ylabel=measure_name, xlabel="episodes")
            subfig.plot(range(len(measure_data)), measure_data)
    for ax in axs.flat:
        ax.label_outer()

    if config.file:
        plot.savefig(config.file)
    else:
        plot.show()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name',type=str,required=True,help="heading of the image")
    parser.add_argument('-f', '--file', type=str,default=None, help="file to write the image to, optional")
    subparser = parser.add_subparsers(title="analyzers",required=True)

    # parser for deep reward analysis
    d_parser = subparser.add_parser("all")
    d_parser.set_defaults(func=visualize_all)
    d_parser.add_argument('result_file',type=argparse.FileType("r"),nargs='+',)

    #parser for simple rewards analysis
    r_parser = subparser.add_parser("single")
    r_parser.set_defaults(func=visualize_single)
    r_parser.add_argument('-m','--measure',required=True, type=str, help="visualize the measure")
    r_parser.add_argument('-c', '--columns', type=int, default=3, help="number of columns to display",)
    r_parser.add_argument('result_file',type=argparse.FileType("r"),nargs='+',)

    return parser.parse_args()
    

def main():
    config = parse_config()
    config.func(config)

if __name__ == "__main__":
    main()
