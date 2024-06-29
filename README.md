# SpatioTemporalCausalLearning

A modularized repository for Spatio-Temporal Causal Learning under ADNI database ([https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)).

## Running
Put ADNI data folder outside this repository like the structure below:
```
├── data
│   └── ADNI-adhd
│       ├── CN
│       ├── ...
│       ├── SMC
│       └── label.csv
└── SpatioTemporalCausalLearning
```
Example: Run NOTEARS
```
cd SpatioTemporalCausalLearning/Baselines/NOTEARS
mkdir ECNs_results
nohup python main.py
```

## Contained Reproducible Baselines
- [NOTEARS] Zheng X, Aragam B, Ravikumar P K, et al. Dags with no tears: Continuous optimization for structure learning[J]. Advances in Neural Information Processing Systems, 2018, 31. [https://github.com/xunzheng/notears](https://github.com/xunzheng/notears)

- [DAGGNN] Yu Y, Chen J, Gao T, et al. DAG-GNN: DAG structure learning with graph neural networks[C]. International Conference on Machine Learning. PMLR, 2019: 7154-7163. [https://github.com/fishmoon1234/DAG-GNN](https://github.com/fishmoon1234/DAG-GNN)

- [NRI] Kipf T, Fetaya E, Wang K C, et al. Neural relational inference for interacting systems[C]. International Conference on Machine Learning. PMLR, 2018: 2688-2697. [https://github.com/ethanfetaya/NRI](https://github.com/ethanfetaya/NRI)

