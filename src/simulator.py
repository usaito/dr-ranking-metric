from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from evaluator import Evaluator
from preprocessor import preprocess_yahoo_coat, preprocess_movielens


def lightfm_trainer(
    train: np.ndarray, loss: str, n_components: int, lam: float
) -> None:
    """Train lightfm models."""
    model = LightFM(
        loss=loss,
        user_alpha=lam,
        item_alpha=lam,
        no_components=n_components,
        learning_rate=0.001,
        random_state=12345,
    )
    dataset = Dataset()
    dataset.fit(train[:, 0], train[:, 1])
    (interactions, weights) = dataset.build_interactions(
        ((x[0], x[1], 1) for x in train[train[:, 2] == 1])
    )
    model.fit(interactions, epochs=100)

    return model


class Simulator:
    def __init__(
        self, data: str, num_sims: int, bounds: List[float], power: float
    ) -> None:
        """Initialize Class."""
        self.data = data
        self.num_sims = num_sims
        self.power = power
        self.bounds = bounds
        self.path = Path(f"../results/{self.data}")
        self.path.mkdir(parents=True, exist_ok=True)
        np.random.seed(12345)

    def run_sims(self, params_list: List) -> None:
        """Run simulations."""
        assert self.num_sims is not None
        if self.data == "ml-100k":
            assert self.power is not None
            self._run_movielens(
                num_sims=self.num_sims, power=self.power, params_list=params_list
            )
        else:
            self._run_yahoo_coat(num_sims=self.num_sims, params_list=params_list)

    def _run_movielens(self, num_sims: int, power: float, params_list: List) -> None:
        """Run semi-synthetic simulations on movie lens 100K."""
        evaluator = Evaluator(data=self.data, bound=self.bounds, k=[5, 10])
        rel_rmse_dict = {
            met: {est: [] for est in evaluator.est_list}
            for met in evaluator.metric_list
        }
        for seed in np.arange(num_sims):
            train, val, test = preprocess_movielens(power=power, seed=seed)
            gt_dict = {met: [] for met in evaluator.metric_list}
            est_dict = {
                met: {est: [] for est in evaluator.est_list}
                for met in evaluator.metric_list
            }
            for params_ in params_list:
                # recommender training
                model = lightfm_trainer(train=train, **params_)
                # recommender evaluation by estimators
                est_results = evaluator.estimate_performance(
                    model=model, val=val, seed=seed
                )
                gt_results = evaluator.evaluate_ground_truth(model=model, test=test)
                # calc performance of estimators
                for metric in evaluator.metric_list:
                    gt_dict[metric].append(gt_results.loc[metric, "gt"])
                    for est in est_results.columns:
                        est_dict[metric][est].append(est_results.loc[metric, est])
            # calc performance of estimators
            for metric in evaluator.metric_list:
                gt_arr = np.array(gt_dict[metric])
                for est in est_results.columns:
                    est_arr = np.array(est_dict[metric][est])
                    rel_rmse_dict[metric][est].append(
                        np.sqrt(np.mean(((est_arr - gt_arr) / gt_arr) ** 2))
                    )

        rel_rmse_df = pd.DataFrame(
            index=list(rel_rmse_dict.keys()), columns=evaluator.est_list
        )
        rel_rel_rmse_df = pd.DataFrame(
            index=list(rel_rmse_dict.keys()), columns=evaluator.est_list
        )
        for metric in evaluator.metric_list:
            baseline = np.mean(rel_rmse_dict[metric]["ips"])
            for est in est_results.columns:
                results_ = rel_rmse_dict[metric][est]
                rel_rmse_df.loc[metric, est] = np.mean(results_)
                rel_rel_rmse_df.loc[metric, est] = np.mean(results_) / baseline
        rel_rmse_df.to_csv(self.path / f"rel_rmse_power={power}.csv")
        rel_rel_rmse_df.to_csv(self.path / f"rel_rel_rmse_power={power}.csv")

    def _run_yahoo_coat(self, num_sims: int, params_list: List) -> None:
        """Run simulations on Coat and Yahoo! R3."""
        evaluator = Evaluator(data=self.data, k=[5, 10, 50])
        est_list = ["naive", "ips", "dr"]
        rel_rmse_dict = {
            met: {est: [] for est in est_list} for met in evaluator.metric_list
        }
        for seed in np.arange(num_sims):
            train, val, test = preprocess_yahoo_coat(data=self.data, seed=seed)
            gt_dict = {met: [] for met in evaluator.metric_list}
            est_dict = {
                met: {est: [] for est in est_list} for met in evaluator.metric_list
            }
            for params_ in params_list:
                # recommender training
                model = lightfm_trainer(train=train, **params_)
                # recommender evaluation by estimators
                est_results = evaluator.estimate_performance(model=model, val=val)
                gt_results = evaluator.evaluate_ground_truth(model=model, test=test)
                # save results by estimators
                for metric in evaluator.metric_list:
                    gt_dict[metric].append(gt_results.loc[metric, "gt"])
                    for est in est_results.columns:
                        est_dict[metric][est].append(est_results.loc[metric, est])
            # calc performance of estimators
            for metric in evaluator.metric_list:
                gt_arr = np.array(gt_dict[metric])
                for est in est_results.columns:
                    est_arr = np.array(est_dict[metric][est])
                    rel_rmse_dict[metric][est].append(
                        np.sqrt(np.mean(((est_arr - gt_arr) / gt_arr) ** 2))
                    )

        # summarize and save performances of estimators
        rel_rmse_df = pd.DataFrame(index=list(rel_rmse_dict.keys()), columns=est_list)
        for metric in evaluator.metric_list:
            for est in est_results.columns:
                # rel-rmse in model evaluation
                results_ = rel_rmse_dict[metric][est]
                stderr = (np.std(results_) / np.sqrt(num_sims)).round(4)
                rel_rmse_df.loc[
                    metric, est
                ] = f" {np.mean(results_).round(4)} ({stderr}) "
        rel_rmse_df.to_csv(self.path / f"rel_rmse.csv")

