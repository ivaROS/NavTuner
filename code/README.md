# navigation_test
This is the local planner navigation testbench for IVALab. This readme aims to guide you to test your own local planner that is implemented as a ```nav_core:BaseLocalPlanner``` plugin to the ```move_base``` navigation stack.
There are two packages under this repository, ```nav_scripts``` and ```nav_configs```. ```nav_scripts``` are the package that initiates the testing. ```nav_configs``` contains the supporting files for ```nav_scripts```, including robot and sensor models, Gazebo world files and launch files for launching Gazebos.
# How to Test Against your Own local planner
1. Clone this repository into your catkin workspace and ```catkin_make```. The latest branch to clone is ```chapter``` branch. You would also need to clone ```gazebo_utils``` to make this work.
2. Prepare the launch file for your local planner. The launch files should launch move_base, include necessary files and launch any necessary node required for your local planner other than everything provided by move_base.
Naming format should follow ```{robotmodel}_{local_planner_name}_controller.launch```, for example, ```turtlebot_teb_controller.launch```. Put your launch file in ```navigation_test/scripts/launch```
3. Find ```navigation_test/scripts/scripts/gazebo_master_test.py``` Scroll all the way to bottom. Fill out test_scenarios and test_controllers with the scenarios and controllers you want to test with. Run the file using python2.
4. There is default to be no Gazebo GUI, if you want to see exact testing details from Gazebo GUI, go to ```nav_configs/launch``` and find the launch file that launches the specific world. In the format of ```gazebo_{robot_name}_{world_name}_world.launch```
