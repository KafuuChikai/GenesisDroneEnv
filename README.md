# Genesis Drone Environment
This repository contains example RL environment for Drones. This project has been included in the **official** [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) repository and [documentation](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hover_env.html). This repository will continue to provide examples and may be **updated with more complex features** in the future.

## Requirements

Ensure you have installed the latest version of [Genesis](https://github.com/Genesis-Embodied-AI/Genesis):

```bash
# Clone the Genesis repository
cd /your/path/to/store/theRepo/
git clone https://github.com/Genesis-Embodied-AI/Genesis.git

# Install Genesis
cd Genesis
pip install -e .
```

Installed all necessary dependencies.

```bash
# Install Requirements.
git submodule update --init --recursive
pip install -r requirements.txt
```

## Demo

If you prefer, you can skip the training and proceed directly using the provided checkpoint `logs/drone-demo/model_500.pt`.

```bash
python hover_eval.py -e drone-demo --ckpt 500 --record
```

## Training

Use the provided training script to start training the policy.

```bash
python hover_train.py -e drone-hovering -B 8192 --max_iterations 300
```

- `-e drone-hovering`: Specifies the experiment name as “drone-hovering”.
- `-B 8192`: Sets the number of environments to 8192 for parallel training.
- `--max_iterations 500`: Specifies the maximum number of training iterations to 500. 
- `-v`: Optional. Enables visualization during training.

If you enable the visualization, you will see:

<img src="docs/training.gif" alt="training" width="70%"/>

To monitor the training process, launch `TensorBoard`:

```bash
tensorboard --logdir logs
```

## Evaluation

### Evaluate the train policy
Use the provided evaluation script to evaluate the trained policy.

```bash
python hover_eval.py -e drone-hovering --ckpt 300 --record
```

- `-e drone-hovering`: Specifies the experiment name as “drone-hovering”.
- `--ckpt 300`: Loads the trained policy from checkpoint 300.
- `--record`: Optional. Records the evaluation and saves a video of the drone’s performance.

The evaluation script will visualize the drone’s performance and save a video if the `--record` flag is set.

<img src="docs/evaluation.gif" alt="evaluation" width="70%"/>

By following this tutorial, you’ll be able to train and evaluate a basic drone hovering policy using Genesis. Have fun and enjoy!