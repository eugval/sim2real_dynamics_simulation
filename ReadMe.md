# Bridging the Gap: A Deep Dive into Zero-Shot Sim-to-Real Transfer for Dynamics

This Repository contains the code used for training the manipulation policies in simulation,
before deploying it ot the real world using the ros package found here: https://github.com/eugval/sim2real_dynamics_robot

### Dependencies
In order to run this project,  the following dependencies need to be satisfied.

Mujoco : http://www.mujoco.org/ \
Openai gym : https://github.com/openai/gym \
Openai mujoco-py : https://github.com/openai/mujoco-py \
urdf_parser_py : https://github.com/ros/urdf_parser_py \
PyKDL : If the pip version does not work, it will need to be compiled from scratch by following the instructions
at https://github.com/orocos/orocos_kinematics_dynamics/issues/115


The following forked repositories:
kdl_parser, the branch fixing_setup_py from https://github.com/eugval/kdl_parser/tree/fixing_setup_py 
simple-pid, the branch allow arrays from https://github.com/eugval/simple-pid/tree/allow_arrays

Other dependencies are listed in  requirements.txt.

### Project Structure
``robosuite-extra`` implements extra environments and functionalities for the robosuite framework

``sim2real-policies`` implements the methods for training and evaluating the different sim2real polcies.

``reality-calibration-characterisation`` implements code to identify and characterise the different sources of error
when transferring from simulation to the real world for the given tasks, as well as scripts to optimise  simulator 
parameters.

