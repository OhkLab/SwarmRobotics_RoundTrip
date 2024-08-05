# SwarmRbotics_RoundTrip
Round trip task for robotic swarm with Unity.

## Version
* python: 3.8.12

## Requirements
```shell
# Create virtual environment of python
cd ml-agents-envs
python -m venv venv
source venv/bin/activate

# Install packages.
pip install -e .
pip install -r requirements.txt
```

## Evaluation
We provide pre-trained policy in the `ml-agents-envs/pretrained` folder.
```shell
# Test the policy.
python eval_agent.py --load-model=pretrained/model.npz
```
**argments**  
`--app-path` : Path to UnityApplication.  
`--load-model` : Path to Model.  
`--timesteps` : Timesteps per episode.  
`--reps` : Number of episode.

## Training
It can take longer to train the agents on a local machine, however it is possible to tune `-n`, `--t`, etc to speed up.
```shell
#Train agent.
mpirun -n 18 --oversubscribe python train_agent.py --t 2 --reps 1
```
**argments**  
`--t` : Roop number of process (population size is `-n` * `--t`).  
`--app-path` : Path to UnityApplication.  
`--log-dir` : Directory of logs.  
`--timesteps` : Timesteps per episode.  
`--max-iter` : Max training iterations.  
`--save-interval` : Model saving period.  
`--seed` : Random seed for evaluation.  
`--reps` : Number of rollouts for fitness.  
`--algo-number` : Index of Evolution strategy (0: CMA-ES, 1: SNES).  
`--init-sigma` : Initial std.  

