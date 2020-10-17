import codecs
import yaml
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.framework import ops

from model import MF


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Calculate sigmoid."""
    return 1 / (1 + np.exp(-x))


def transform_rating(ratings: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Transform ratings into graded relevance information."""
    ratings -= 1
    return eps + (1.0 - eps) * (2 ** ratings - 1) / (2 ** np.max(ratings) - 1)


def preprocess_movielens(
    power: float = 1.0, seed: int = 12345
) -> Dict[str, np.ndarray]:
    """Load and preprocess ML 100K."""
    np.random.seed(seed)

    with open("../config.yaml", "rb") as f:
        config = yaml.safe_load(f)
        val_size = config["val_size"]
        hyperparams = config["mf_hyperparams"]

    with codecs.open(
        f"../data/ml-100k/ml-100k.data", "r", "utf-8", errors="ignore"
    ) as f:
        data = pd.read_csv(f, delimiter="\t", header=None).loc[:, :2]
        data.rename(columns={0: "user", 1: "item", 2: "rate"}, inplace=True)
        data.user, data.item = data.user - 1, data.item - 1
        data = data.values

    num_users, num_items = data[:, 0].max() + 1, data[:, 1].max() + 1
    user_item_ = (
        pd.DataFrame(np.zeros((num_users, num_items)))
        .stack()
        .reset_index()
        .values[:, :2]
    )
    # generate CVR by MF.
    ops.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(seed)
    model = MF(
        num_users=num_users,
        num_items=num_items,
        dim=hyperparams["dim"],
        eta=hyperparams["eta"],
        lam=hyperparams["lam"],
    )
    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for _ in np.arange(hyperparams["iters"]):
        idx = np.random.choice(np.arange(data.shape[0]), size=hyperparams["batch_size"])
        _ = sess.run(
            model.apply_grads_mse,
            feed_dict={
                model.users: data[idx, 0],
                model.items: data[idx, 1],
                model.labels: np.expand_dims(data[idx, 2], 1),
                model.pscore: np.ones((hyperparams["batch_size"], 1)),
            },
        )
    cvr = sess.run(
        model.preds,
        feed_dict={model.users: user_item_[:, 0], model.items: user_item_[:, 1]},
    )
    cvr = np.clip(cvr.flatten(), 1, 5)
    cvr = transform_rating(cvr, eps=0.1)
    cv = np.random.binomial(n=1, p=cvr)

    # generate CTR by logistic MF.
    all_data = (
        pd.DataFrame(np.zeros((num_users, num_items)))
        .stack()
        .reset_index()
        .values[:, :2]
    )
    pos_data = data[:, :2]
    unlabeled_data = np.array(
        list(set(map(tuple, all_data)) - set(map(tuple, pos_data))), dtype=int
    )
    data = np.r_[
        np.c_[pos_data, np.ones(pos_data.shape[0])],
        np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])],
    ]

    ops.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(seed)
    model = MF(
        num_users=num_users,
        num_items=num_items,
        dim=hyperparams["dim"],
        eta=hyperparams["eta"],
        lam=hyperparams["lam"],
    )
    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for _ in np.arange(hyperparams["iters"]):
        idx = np.random.choice(np.arange(data.shape[0]), size=hyperparams["batch_size"])
        _ = sess.run(
            model.apply_grads_ce,
            feed_dict={
                model.users: data[idx, 0],
                model.items: data[idx, 1],
                model.labels: np.expand_dims(data[idx, 2], 1),
                model.pscore: np.ones((hyperparams["batch_size"], 1)),
            },
        )
    ctr = sess.run(
        model.preds,
        feed_dict={model.users: user_item_[:, 0], model.items: user_item_[:, 1]},
    )
    ctr = sigmoid(ctr.flatten()) ** power
    ct = np.random.binomial(n=1, p=ctr)

    train_indicator = np.random.binomial(n=1, p=(1.0 - val_size), size=ct.shape[0])
    ct_train, ct_val = ct * train_indicator, ct * (1 - train_indicator)
    train = np.c_[user_item_, ct_train * cv]
    val = np.c_[user_item_, ct_val * cv, ct_val, cv, ctr * val_size, cvr]
    test = np.c_[user_item_, ct * cv, ct, cv, ctr, cvr]

    return train, val, test


def preprocess_yahoo_coat(
    data: str, val_ratio: float = 0.3, seed: int = 12345
) -> Tuple:
    """Load and preprocess Yahoo! R3 and Coat datasets."""
    np.random.seed(seed)

    with open("../config.yaml", "rb") as f:
        hyperparams = yaml.safe_load(f)["mf_hyperparams"]

    if data == "yahoo":
        cols = {0: "user", 1: "item", 2: "rate"}
        with codecs.open(
            f"../data/yahoo/train.txt", "r", "utf-8", errors="ignore"
        ) as f:
            train_ = pd.read_csv(f, delimiter="\t", header=None)
            train_.rename(columns=cols, inplace=True)
        with codecs.open(f"../data/yahoo/test.txt", "r", "utf-8", errors="ignore") as f:
            test_ = pd.read_csv(f, delimiter="\t", header=None)
            test_.rename(columns=cols, inplace=True)
        for data_ in [train_, test_]:
            data_.user, data_.item = data_.user - 1, data_.item - 1
    elif data == "coat":
        cols = {"level_0": "user", "level_1": "item", 2: "rate", 0: "rate"}
        with codecs.open(
            f"../data/coat/train.ascii", "r", "utf-8", errors="ignore"
        ) as f:
            train_ = pd.read_csv(f, delimiter=" ", header=None)
            train_ = train_.stack().reset_index().rename(columns=cols)
            train_ = train_[train_.rate != 0].reset_index(drop=True)
        with codecs.open(
            f"../data/coat/test.ascii", "r", "utf-8", errors="ignore"
        ) as f:
            test_ = pd.read_csv(f, delimiter=" ", header=None)
            test_ = test_.stack().reset_index().rename(columns=cols)
            test_ = test_[test_.rate != 0].reset_index(drop=True)
    # binarize ratings
    for data_ in [train_, test_]:
        data_.rate = np.array(data_.rate >= 4, dtype=int)
    # estimate propensity score by MF
    train, test = train_.values, test_.values
    pos_train = train_[train_.rate == 1].values
    pos_test = test_[test_.rate == 1].values
    # preprocess datasets
    unique_user_train, user_counts_train = np.unique(
        pos_train[:, 0], return_counts=True
    )
    unique_user_train = unique_user_train[user_counts_train >= 2]
    unique_user_test, user_counts_test = np.unique(pos_test[:, 0], return_counts=True)
    unique_user_test = unique_user_test[user_counts_test <= 9]
    valid_users = np.intersect1d(unique_user_train, unique_user_test)
    train = train[np.array([u in valid_users for u in train[:, 0]])]
    test = test[np.array([u in valid_users for u in test[:, 0]])]
    train[:, 0] = stats.rankdata(train[:, 0], method="dense") - 1
    test[:, 0] = stats.rankdata(test[:, 0], method="dense") - 1

    num_users, num_items = train[:, 0].max() + 1, train[:, 1].max() + 1
    all_data = (
        pd.DataFrame(np.zeros((num_users, num_items)))
        .stack()
        .reset_index()
        .values[:, :2]
    )
    unobs_data = np.array(
        list(set(map(tuple, all_data)) - set(map(tuple, train[:, :2])))
    )
    train = np.r_[
        np.c_[train, np.ones(train.shape[0])],
        np.c_[unobs_data, np.zeros((unobs_data.shape[0], 2))],
    ]
    train, val = train_test_split(train, test_size=val_ratio, random_state=seed)
    unobs_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, val[:, :2]))))
    val = np.r_[val, np.c_[unobs_data, np.zeros((unobs_data.shape[0], 2))]]

    # define the matrix factorization model
    ops.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(seed)
    model = MF(
        num_users=num_users,
        num_items=num_items,
        dim=hyperparams["dim"],
        eta=hyperparams["eta"],
        lam=hyperparams["lam"],
    )
    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for _ in np.arange(hyperparams["iters"]):
        idx = np.random.choice(np.arange(val.shape[0]), size=hyperparams["batch_size"])
        _ = sess.run(
            model.apply_grads_ce,
            feed_dict={
                model.users: val[idx, 0],
                model.items: val[idx, 1],
                model.labels: np.expand_dims(val[idx, 3], 1),
                model.pscore: np.ones((hyperparams["batch_size"], 1)),
            },
        )
    # obtain dense user-item matrix
    ctr_hat = sess.run(
        model.preds,
        feed_dict={
            model.users: val[:, 0].astype(int),
            model.items: val[:, 1].astype(int),
        },
    )
    val = np.c_[val, sigmoid(ctr_hat)]

    # estimate relevance parameter (gamma) by MF.
    ops.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(seed)
    model = MF(
        num_users=num_users,
        num_items=num_items,
        dim=hyperparams["dim"],
        eta=hyperparams["eta"],
        lam=hyperparams["lam"],
    )
    # observed data
    val_obs = val[val[:, 3] == 1]
    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for _ in np.arange(hyperparams["iters"]):
        idx = np.random.choice(
            np.arange(val_obs.shape[0]), size=hyperparams["batch_size"]
        )
        _ = sess.run(
            model.apply_grads_ce,
            feed_dict={
                model.users: val_obs[idx, 0],
                model.items: val_obs[idx, 1],
                model.labels: np.expand_dims(val_obs[idx, 2], 1),
                model.pscore: np.expand_dims(val_obs[idx, 4], 1),
            },
        )
    # obtain dense user-item matrix
    gamma_hat = sess.run(
        model.preds,
        feed_dict={
            model.users: val[:, 0].astype(int),
            model.items: val[:, 1].astype(int),
        },
    )
    val = np.c_[val, sigmoid(gamma_hat)]
    # create test data containing all items
    all_data = (
        pd.DataFrame(np.zeros((num_users, num_items)))
        .stack()
        .reset_index()
        .values[:, :2]
    )
    unobs_data = np.array(
        list(set(map(tuple, all_data)) - set(map(tuple, test[:, :2])))
    )
    test = np.r_[
        np.c_[test, np.ones(test.shape[0])],
        np.c_[unobs_data, np.zeros((unobs_data.shape[0], 2))],
    ]
    avg_test_pscore = test[:, -1].mean()
    test = np.c_[test, np.ones(test.shape[0]) * avg_test_pscore]

    return train, val, test
