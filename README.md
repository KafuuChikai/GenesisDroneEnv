# Genesis Drone Environment
This repository contains example RL environment for Drones. This project has been included in the **official** [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) repository and [documentation](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hover_env.html). This repository will continue to provide examples and may be **updated with more complex features** in the future.

## Installations

It's recommended to use a virtual environment, such as conda:

```bash
conda create -n genesis_drone python=3.11 # Requires Python >= 3.10, <3.13
conda activate genesis_drone
```

Ensure you have installed the latest version of [Genesis](https://github.com/Genesis-Embodied-AI/Genesis). And then install the repo:

```bash
git clone https://github.com/KafuuChikai/GenesisDroneEnv.git
pip install -e .
```

## Demo

If you prefer, you can skip the training and proceed directly using the provided checkpoint `logs/drone-demo/model_500.pt`.

```bash
python scripts/hover_eval.py -e drone-demo --ckpt 500 --record
```

## Training

Use the provided training script to start training the policy.

```bash
python scripts/hover_train.py -e drone-hovering -B 8192 --max_iterations 300
```

- `-e drone-hovering`: Specifies the experiment name as “drone-hovering”.
- `-B 8192`: Sets the number of environments to 8192 for parallel training.
- `--max_iterations 300`: Specifies the maximum number of training iterations to 300. 
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
python scripts/hover_eval.py -e drone-hovering --ckpt 300 --record
```

- `-e drone-hovering`: Specifies the experiment name as “drone-hovering”.
- `--ckpt 300`: Loads the trained policy from checkpoint 300.
- `--record`: Optional. Records the evaluation and saves a video of the drone’s performance.

The evaluation script will visualize the drone’s performance and save a video if the `--record` flag is set.

https://github.com/user-attachments/assets/dc3f8b50-4bf6-4ad7-b6f3-822157190073

By following this tutorial, you’ll be able to train and evaluate a basic drone hovering policy using Genesis. Have fun and enjoy!

## Acknowledgement

This repository is inspired by the following work:

1. [Champion-level drone racing using deep reinforcement learning (Nature 2023)](https://www.nature.com/articles/s41586-023-06419-4.pdf)

We acknowledge the contributions of these and future works that inspire and enhance the development of this repository.
