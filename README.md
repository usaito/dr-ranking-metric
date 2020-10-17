## Doubly Robust Estimator for Ranking Metrics with Post-Click Conversions

---

### About

This repository contains the code for the real-world experiment conducted in the paper [Doubly Robust Estimator for Ranking Metrics with Post-Click Conversions](https://dl.acm.org/doi/abs/10.1145/3383313.3412262) by [Yuta Saito](https://usaito.github.io/), which has been accepted to [RecSys2020](https://recsys.acm.org/recsys20/).

If you find this code useful in your research then please cite:
```
@inproceedings{saito2020doubly,
author = {Saito, Yuta},
title = {Doubly Robust Estimator for Ranking Metrics with Post-Click Conversions},
year = {2020},
booktitle = {Fourteenth ACM Conference on Recommender Systems},
pages = {92–100},
location = {Virtual Event, Brazil},
series = {RecSys '20}
}
```

### Dependencies

- numpy==1.19.1
- pandas==1.1.2
- scikit-learn==0.23.1
- tensorflow==1.15.4
- lightfm==1.15.0

### Datasets
To run the simulations, the following datasets need to be prepared as described below.

- download the [Yahoo! R3 dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r) and put `train.txt` and `test.txt` files into `./data/yahoo/` directory.
- download the [Coat dataset](https://www.cs.cornell.edu/~schnabts/mnar/) and put `train.ascii` and `test.ascii` files into `./data/coat/` directory.

### Running the code

To run the real-world experiment, navigate to the `src/` directory and run the following commands

```bash
python main.py --num_sims 5 --data coat
```

```bash
python main.py --num_sims 20 --data yahoo
```

Once the code is finished executing, you can find the summarized results (relative-RMSEs, lower value is better) in `./results/` directory.
