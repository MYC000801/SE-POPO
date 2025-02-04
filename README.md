
# Avoiding $\exp(R_{max})$ scaling in RLHF through Preference-based Exploration

This repository contains the code and released models for algorithm **SE-POPO** in https://arxiv.org/abs/2502.00666. 

This codebase is mainly based on Online-RLHF https://github.com/RLHFlow/Online-RLHF

### 1. Create Conda Envs

```
conda env create -f conda_envs/vllm.yml
conda env create -f conda_envs/rlhflow.yml
```

### 2. Run bash file

```
bash LLama_SFT_wpo_1e2.sh
```

