import yaml
import argparse
import warnings

from simulator import Simulator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", "-d", type=str, required=True, choices=["ml-100k", "yahoo", "coat"]
)
parser.add_argument("--num_sims", "-n", type=int, default=1)
parser.add_argument("--power", "-p", type=float, default=1.0)
parser.add_argument(
    "--bound_list", "-b", type=list, default=[0.01, 0.05, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]
)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    # configurations
    data = args.data
    num_sims = args.num_sims
    power = args.power
    bounds = args.bound_list

    # search space
    config = yaml.safe_load(open("../config.yaml", "r"))["search_space"]
    dim_list = config["dim"]
    lam_list = config["lam"]
    loss_list = config["loss"]

    params_list = []
    for loss in loss_list:
        for n_components in dim_list:
            for lam in lam_list:
                params = {}
                params["loss"] = loss
                params["n_components"] = n_components
                params["lam"] = lam
                params_list.append(params)

    sim = Simulator(data=data, num_sims=num_sims, bounds=bounds, power=power)
    sim.run_sims(params_list=params_list)
