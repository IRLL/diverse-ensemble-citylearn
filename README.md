# diverse-ensemble-citylearn
The codebase replicating the results of the "Ensembling Diverse Policies Improves Generalizability of Reinforcement Learning Algorithms in Continuous Control Tasks" paper and similarly titled thesis work.

# Installation
Python 3.9 is recommended
```
pip install -r requirements.txt
pip install citylearn==1.3.6 --no-deps
```

# Running an experiment
```
python train.py --wandb-entity <wandb team/user name>
```