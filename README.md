# README

## Setting up Conda environment and packages
Create a new conda environment and go to the home directory (mushr_push_sim: `$HOME`)

```bash
conda create -n my_env python=3.7
conda activate my_env
cd $HOME
pip install -r requirements.txt
```

## Data Generation
To generate Data, you need a valid license of MuJoCo and must be able to run the Figure 8 tutorial on the Mushr website (https://mushr.io/tutorials/mujoco_figure8/).

```bash
rm -f $HOME/datacol/json_files/*
python collect_data.py --cpu_count 3
```
This command will generate train.csv.gz, test_seen.csv.gz and test_unseen.csv.gz in `$HOME/datacol/json_files/`.

## Training the Model
Note: You might want to train the model on a server since the dataloader consumer a lot of memory, which might hang your laptop/computer, and it is preferred to have a GPU (also I haven't tested the code on cpu, but it should work). We first need to set the `data_dir`, `home_dir`, `traj_save_addr`, `model_path` and `train` variables in `$HOME/train/main.py`. You can set these values by passing them as arguments or change it in the code itself.

`data_dir` is the parent folder of data stored (`$HOME/datacol/json_files`).

`home_dir` is the parent folder for main script file (`$HOME/train/main.py`).

`traj_save_addr` is the parent folder where you want to save your visual trajectories.

`model_path` is the folder where you want to store your pytorch model.

`train` is basically True when you want to train a model. If you simply want to evaluate and visualize an already pretrained lstm or simple regression model, then you can disable train. This comes handy when you want to evaluate your model and visualize trajectories, you dont have to train all over again.

Note: Dont use '~/' notation for paths, rather mention the complete path. For example: '/home/user/'.

Now run the training code as follows:
```bash
cd $HOME/train/
python main.py --data_dir '{data_dir}' --home_dir '{home_dir}' --traj_save_addr '{traj_save_addr}'
```
If you made the changes in the argument parser of the code simply run,
```bash
cd $HOME/train/
python main.py
```

## Visualizations
There are a total of three visuals, the trajectories, red are predicted positions and blue the actual positions of the block stored in `{traj_save_addr}`, the tensorboard visualizations of the train and test loss plots in `$HOME/train/TensorboardVisuals/`, and finally Average Trajectory MSE losses stored in the current directory (during code execution).
To run the tensorboard plots, simple run in terminal
```bash
cd $HOME/train/
tensorboard --logdir=TensorboardVisuals/ --bind_all
```

To visualize the model,
```bash
cd $HOME/train/
python -c "import netron; netron.start('model.onnx');"
```

The trajectory error plots can be found in the `$HOME/train/` folder. They consist of absolute errors in x,y and rotation about z coordinates (theta) per trajectory. The index variable in the x-axis is the index of point in the trajectory.

Absolute error plots.
<p float="left" align="middle">
	<img src="/Images/x.png" width=400 title="Absolute error in x" />
	<img src="/Images/y.png" width=400 title="Absolute error in y" />
	<img src="/Images/theta.png" width=400 title="Absolute error in theta" />
</p> 
