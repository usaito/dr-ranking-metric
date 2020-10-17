from typing import List, Optional
from lightfm import LightFM

import numpy as np
import pandas as pd


def dcg_at_k(
    ct: np.ndarray,
    cv: np.ndarray,
    score: np.ndarray,
    k: int,
    cv_hat: Optional[np.array] = None,
    pscore: Optional[np.array] = None,
) -> float:
    """Calculate DCG score."""
    sort_key = score.argsort()[::-1]
    ct_sorted = ct[sort_key]
    cv_sorted = cv[sort_key]
    cv_hat_sorted = cv_hat[sort_key] if cv_hat is not None else np.zeros_like(cv)
    pscore_sorted = pscore[sort_key] if pscore is not None else np.ones_like(cv)

    dcg_score = cv_hat_sorted[0]
    dcg_score += ct_sorted[0] * (cv_sorted[0] - cv_hat_sorted[0]) / pscore_sorted[0]
    dcg_score_ = cv_hat_sorted[1:k]
    dcg_score_ += (
        ct_sorted[1:k] * (cv_sorted[1:k] - cv_hat_sorted[1:k]) / pscore_sorted[1:k]
    )
    dcg_score += (dcg_score_ / np.log2(np.arange(1, k) + 1)).sum()
    denominator = (
        cv_hat_sorted + ct_sorted * (cv_sorted - cv_hat_sorted) / pscore_sorted
    ).sum()
    final_score = np.clip(dcg_score / denominator, 0, 1) if denominator != 0 else 0.0

    return final_score


def recall_at_k(
    ct: np.ndarray,
    cv: np.ndarray,
    score: np.ndarray,
    k: int,
    cv_hat: Optional[np.array] = None,
    pscore: Optional[np.array] = None,
) -> float:
    """Calculate recall score."""
    sort_key = score.argsort()[::-1]
    ct_sorted = ct[sort_key]
    cv_sorted = cv[sort_key]
    cv_hat_sorted = cv_hat[sort_key] if cv_hat is not None else np.zeros_like(cv)
    pscore_sorted = pscore[sort_key] if pscore is not None else np.ones_like(cv)

    recall = cv_hat_sorted[:k].sum()
    recall += (
        ct_sorted[:k] * (cv_sorted[:k] - cv_hat_sorted[:k]) / pscore_sorted[:k]
    ).sum()
    denominator = (
        cv_hat_sorted + ct_sorted * (cv_sorted - cv_hat_sorted) / pscore_sorted
    ).sum()
    final_score = np.clip(recall / denominator, 0, 1) if denominator != 0 else 0

    return final_score


class Evaluator:

    metrics = {"DCG": dcg_at_k, "Recall": recall_at_k}

    def __init__(self, data: str, k: List, bound: List = None) -> None:
        """Initialize class."""
        self.k = k
        self.data = data
        self.bound = bound
        if self.bound:
            self.est_list = ["naive", "ips"] + [f"dr-{u}" for u in bound]
        self.metric_list = [f"{metric}@{_k}" for metric in self.metrics for _k in k]

    def estimate_performance(
        self, model: LightFM, val: np.ndarray, seed: int = 0
    ) -> None:
        """Estimate the performance of a recommender."""
        if "ml" in self.data:
            return self._estimate_performance_movielens(model=model, val=val, seed=seed)
        else:
            return self._estimate_performance_yahoo_coat(model=model, val=val)

    def evaluate_ground_truth(self, model: LightFM, test: np.ndarray) -> None:
        """Evaluate a recommender by the ground truth perference labels."""
        results = {}

        users = test[:, 0].astype(int)
        items = test[:, 1].astype(int)
        cv = np.zeros(test.shape[0]) if "ml" in self.data else test[:, 2]
        cvr = test[:, -1] if "ml" in self.data else np.zeros(test.shape[0])
        ct = np.zeros(test.shape[0]) if "ml" in self.data else test[:, 3]

        for _k in self.k:
            for metric in self.metrics:
                results[f"{metric}@{_k}"] = []

        for user in set(users):
            indices = users == user
            items_for_current_user = items[indices]
            cvr_for_current_user = cvr[indices]
            ct_for_current_user = ct[indices]
            cv_for_current_user = cv[indices]

            # predict ranking score for each user
            scores = model.predict(
                user_ids=np.int(user), item_ids=items_for_current_user
            )
            # calculate ranking metrics
            for _k in self.k:
                for metric, metric_func in self.metrics.items():
                    results[f"{metric}@{_k}"].append(
                        metric_func(
                            cv=cv_for_current_user,
                            ct=ct_for_current_user,
                            cv_hat=cvr_for_current_user,
                            score=scores,
                            k=_k,
                        )
                    )
        # aggregate results
        gt_results = pd.DataFrame(index=results.keys())
        gt_results["gt"] = list(map(np.mean, list(results.values())))
        return gt_results

    def _estimate_performance_movielens(
        self, model: LightFM, val: np.ndarray, seed: int = 0
    ) -> None:
        """Estimate the performance of a recommender with semi-synthetic data."""
        naive_results = {}
        ips_results = {}
        dr_results = {}
        for _k in self.k:
            for metric in self.metrics:
                naive_results[f"{metric}@{_k}"] = []
                ips_results[f"{metric}@{_k}"] = []
        for u in self.bound:
            dict_ = {}
            for _k in self.k:
                for metric in self.metrics:
                    dict_[f"{metric}@{_k}"] = []
            dr_results[u] = dict_

        users = val[:, 0].astype(int)
        items = val[:, 1].astype(int)
        cv = val[:, 2]
        ct = val[:, 3]
        cvr = val[:, 6]
        pscore = val[:, 5]

        for user in set(users):
            indices = users == user
            items_for_current_user = items[indices]
            cv_for_current_user = cv[indices]
            ct_for_current_user = ct[indices]
            cvr_for_current_user = cvr[indices]
            pscore_for_current_user = pscore[indices]

            # predict ranking score for each user
            scores = model.predict(
                user_ids=np.int(user), item_ids=items_for_current_user
            )
            # calculate ranking metrics
            for _k in self.k:
                for metric, metric_func in self.metrics.items():
                    # estimate the performance by the naive estimator
                    naive_results[f"{metric}@{_k}"].append(
                        metric_func(
                            cv=cv_for_current_user,
                            ct=ct_for_current_user,
                            score=scores,
                            k=_k,
                        )
                    )
                    # estimate the performance by the inverse propensity estimator
                    ips_results[f"{metric}@{_k}"].append(
                        metric_func(
                            cv=cv_for_current_user,
                            ct=ct_for_current_user,
                            score=scores,
                            k=_k,
                            pscore=pscore_for_current_user,
                        )
                    )
                    # estimate the performance by the doubly robsut estimator
                    for u in self.bound:
                        cv_hat = np.clip(
                            cvr_for_current_user + np.random.uniform(-u, u), 0, 1
                        )
                        dr_results[u][f"{metric}@{_k}"].append(
                            metric_func(
                                cv=cv_for_current_user,
                                ct=ct_for_current_user,
                                score=scores,
                                k=_k,
                                cv_hat=cv_hat,
                                pscore=pscore_for_current_user,
                            )
                        )

        # aggregate estimated results
        est_results = pd.DataFrame(index=naive_results.keys())
        est_results["naive"] = list(map(np.mean, list(naive_results.values())))
        est_results["ips"] = list(map(np.mean, list(ips_results.values())))
        for u in self.bound:
            est_results[f"dr-{u}"] = list(map(np.mean, list(dr_results[u].values())))
        return est_results

    def _estimate_performance_yahoo_coat(self, model: LightFM, val: np.ndarray) -> None:
        """Estimate the performance of a recommender with real-world datasets."""
        naive_results = {}
        ips_results = {}
        dr_results = {}

        for _k in self.k:
            for metric in self.metrics:
                naive_results[f"{metric}@{_k}"] = []
                ips_results[f"{metric}@{_k}"] = []
                dr_results[f"{metric}@{_k}"] = []

        users = val[:, 0].astype(int)
        items = val[:, 1].astype(int)
        cv = val[:, 2]
        ct = val[:, 3]
        cv_hat = val[:, 5]
        pscore = val[:, 4]

        for user in set(users):
            indices = users == user
            items_for_current_user = items[indices]
            cv_for_current_user = cv[indices]
            ct_for_current_user = ct[indices]
            cv_hat_for_current_user = cv_hat[indices]
            pscore_for_current_user = pscore[indices]

            # predict ranking score for each user
            scores = model.predict(
                user_ids=np.int(user), item_ids=items_for_current_user
            )
            # calculate ranking metrics
            for _k in self.k:
                for metric, metric_func in self.metrics.items():
                    # estimate the performance by the naive estimator
                    naive_results[f"{metric}@{_k}"].append(
                        metric_func(
                            cv=cv_for_current_user,
                            ct=ct_for_current_user,
                            score=scores,
                            k=_k,
                        )
                    )
                    # estimate the performance by the inverse propensity estimator
                    ips_results[f"{metric}@{_k}"].append(
                        metric_func(
                            cv=cv_for_current_user,
                            ct=ct_for_current_user,
                            score=scores,
                            k=_k,
                            pscore=pscore_for_current_user,
                        )
                    )
                    # estimate the performance by the doubly robsut estimator
                    dr_results[f"{metric}@{_k}"].append(
                        metric_func(
                            cv=cv_for_current_user,
                            ct=ct_for_current_user,
                            score=scores,
                            k=_k,
                            cv_hat=cv_hat_for_current_user,
                            pscore=pscore_for_current_user,
                        )
                    )
        # aggregate estimated results
        est_results = pd.DataFrame(index=naive_results.keys())
        est_results["naive"] = list(map(np.mean, list(naive_results.values())))
        est_results["ips"] = list(map(np.mean, list(ips_results.values())))
        est_results["dr"] = list(map(np.mean, list(dr_results.values())))
        return est_results

