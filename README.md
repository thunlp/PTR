# PTR


Code and datasets for our paper "PTR: Prompt Tuning with Rules for Text Classification"

If you use the code, please cite the following paper:

```
@article{han2021ptr,
  title={PTR: Prompt Tuning with Rules for Text Classification},
  author={Han, Xu and Zhao, Weilin and Ding, Ning and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2105.11259},
  year={2021}
}
```

Requirements
==========

The model is implemented using PyTorch. The versions of packages used are shown below.


*	numpy>=1.18.0

*	scikit-learn>=0.22.1

*	scipy>=1.4.1

*	torch>=1.3.0

*	tqdm>=4.41.1

*	transformers>=4.0.0


Baselines
==========

Some baselines, especially the baselines using entity markers, come from the project [[RE_improved_baseline]](https://github.com/wzhouad/RE_improved_baseline).

Datasets
==========

We provide all the datasets and prompts used in our experiments.

+ [[TACRED]](../datasets/tacred)

+ [[TACREV]](../datasets/tacrev)

+ [[RETACRED]](../datasets/retacred)


Run the experiments
==========


#### (1) For TACRED

```
mkdir results
cd results
mkdir tacred
cd tacred
mkdir train
mkdir val
mkdir test
cd ..
cd ..
cd code_script
bash run_large_tacred.sh
```

#### (2) For TACREV

```
mkdir results
cd results
mkdir tacrev
cd tacrev
mkdir train
mkdir val
mkdir test
cd ..
cd ..
cd code_script
bash run_large_tacrev.sh
```

#### (3) For RETACRED

```
mkdir results
cd results
mkdir retacred
cd retacred
mkdir train
mkdir val
mkdir test
cd ..
cd ..
cd code_script
bash run_large_retacred.sh
```
