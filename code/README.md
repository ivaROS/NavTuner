# NavTuner
This is the training/testing/evaluation code for NavTuner. This readme aims to guide you through the basic usage of this code. The code is tested with Ubuntu 16.04 and PyTorch 1.0 with CPU.

## Functionality
The basic functionality of this code includes 1> single and manual random test with random start/goal position in a Gazebo world, using ROS ```move_base``` navigation stack, 2> the automated version of 1, and 3> training/testing/evaluating different NavTuners.
There are two packages under this repository, ```nav_scripts``` and ```nav_configs```. ```nav_scripts``` are the package that initiates the testing. ```nav_configs``` contains the supporting files for ```nav_scripts```, including robot and sensor models, Gazebo world files and launch files for launching Gazebos, and the model definition/training/evaluation code for different NavTuners.

## Requirements
### Dependencies
* ROS catkin package
	* ```roscpp```
	* ```rospy```
	* ```cv_bridge```
	* ```image_transport```
	* ```tf2_ros```
	* ```map_server```
	* ```move_base```
	* ```tf```
	* ```turtlebot_bringup```
* python scripts for running manual/automatic experiments
	* ```python==2.7```
	* ```numpy```
	* ```pickle```
* python scripts for running NavTuners
	* ```pytorch```
	* ```scikit-learn```
	* ```tqdm```
	
### Local Planners
To begin with, you'll need a local planner of your choice that is implemented as a ```nav_core:BaseLocalPlanner``` plugin to the ```move_base``` navigation stack. To run our experiments, you'll need the [egoTEB](https://github.com/ivaROS/egoTEB) package.

## How to Test against Different Local Planner/World w/o NavTuner
1. Clone this repository into your catkin workspace and ```catkin_make```.
2. Prepare the launch file for your local planner. The launch files should launch move_base, include necessary files and launch any necessary node required for your local planner other than everything provided by move_base.
Naming format should follow ```{robotmodel}_{local_planner_name}_controller.launch```, for example, ```turtlebot_teb_controller.launch```. Put your launch file in ```scripts/launch```
3. Find ```scripts/scripts/gazebo_master.py``` Scroll all the way to bottom. Fill out test_scenarios and test_controllers with the scenarios and controllers you would like to test with, and comment out everything related to NavTuner. Run the file using python2.
4. There is default to be no Gazebo GUI, if you want to see exact testing details from Gazebo GUI, go to ```configs/launch``` and find the launch file that launches the specific world. In the format of ```gazebo_{robot_name}_{world_name}_world.launch```.

## How to Test against Different Local Planner/World w/ Different NavTuner
1. Complete step 1 and 2 above.
2. Prepare the NavTuners. Go to ```scripts/scripts``` and modify ```gazebo_master.py``` for linear NavTuners, ```deep_networks.py``` for NN/CNN NavTuners, and ```DQN.py``` for RL NavTuners. Linear models are implemented with ```scikit-learn``` while NN/CNN/RL models are implemented with ```pytorch```.
3. Go to ```gazebo_master.py``` for experiments with linear NavTuners, ```gazebo_dl.py``` for NN/CNN NavTuners, and ```gazebo_rl.py``` for RL NavTuners. Fill out envs/scenes, controllers, and random seeds with what you would like to test with. Run the file using python2.
4. There is default to be no Gazebo GUI, if you want to see exact testing details from Gazebo GUI, go to ```configs/launch``` and find the launch file that launches the specific world. In the format of ```gazebo_{robot_name}_{world_name}_world.launch```.

## Notice
1. We limit PyTorch thread number to ```1``` by default, because multi-thread caused problems during our experiment. If you want to enable multi-thread, please comment out all appearance of ```torch.set_num_threads(1)```, and use at your own risk.
2. During our experiments, occasionaly some NavTuners may cause ```move_base``` to crash right after navigation starts. We avoid this in experiments by starting the NavTuners a few seconds (2s in our experiments) later after navigation starts. If you experience ```move_base``` crashes, try increasing this delay.
