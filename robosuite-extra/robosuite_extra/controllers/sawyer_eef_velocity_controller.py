import numpy as np
import sys
import os

if '/opt/ros/melodic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
    sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import PyKDL
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromFile, treeFromUrdfModel
import robosuite_extra.utils.transform_utils as T


class SawyerEEFVelocityController(object):

    def __init__(self):
        self.num_joints = 7
        # Get the URDF
        urdf_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), '../assets/urdf/sawyer_intera.urdf')

        self.urdf_model = URDF.from_xml_string(open(urdf_path).read())

        # Set the base link, and the tip link.
        self.base_link = self.urdf_model.get_root()
        self.tip_link = "right_hand"
        # # Create a KDL tree from the URDF model. This is an object defining the kinematics of the entire robot.
        self.kdl_tree = treeFromUrdfModel(self.urdf_model)
        # Create a KDL chain from the tree. This is one specific chain within the overall kinematics model.
        self.arm_chain = self.kdl_tree[1].getChain(self.base_link, self.tip_link)
        # Create a solver which will be used to compute the forward kinematics
        self.forward_kinematics_solver = PyKDL.ChainFkSolverPos_recursive(self.arm_chain)
        # Create a solver which will be used to compute the Jacobian
        self.jacobian_solver = PyKDL.ChainJntToJacSolver(self.arm_chain)

        #Velocity inverse kinematics solver
        self.IK_v_solver = PyKDL.ChainIkSolverVel_pinv(self.arm_chain)

        # Create a solver to retrieve the joint angles form the eef position
        self.IK_solver = PyKDL.ChainIkSolverPos_NR(self.arm_chain, self.forward_kinematics_solver, self.IK_v_solver)


    def compute_joint_angles_for_endpoint_pose(self, target_endpoint_pose_in_base_frame, q_guess):
        pos_kdl  = PyKDL.Vector(*target_endpoint_pose_in_base_frame[:3,-1])

        rot_kdl = PyKDL.Rotation(target_endpoint_pose_in_base_frame[0,0],target_endpoint_pose_in_base_frame[0,1], target_endpoint_pose_in_base_frame[0,2],
                                 target_endpoint_pose_in_base_frame[1, 0], target_endpoint_pose_in_base_frame[1,1],target_endpoint_pose_in_base_frame[1,2],
                                 target_endpoint_pose_in_base_frame[2, 0], target_endpoint_pose_in_base_frame[2,1], target_endpoint_pose_in_base_frame[2,2])

        frame_kdl = PyKDL.Frame(rot_kdl, pos_kdl)
        q_guess_kdl = self.convert_joint_angles_array_to_kdl_array(q_guess)

        kdl_jnt_array = PyKDL.JntArray(self.num_joints)
        self.IK_solver.CartToJnt(q_guess_kdl,frame_kdl,kdl_jnt_array)

        return self.kdl_1Darray_to_numpy(kdl_jnt_array)


    # Function to compute the robot's joint velocities for a desired Cartesian endpoint (robot gripper) velocity
    def compute_joint_velocities_for_endpoint_velocity(self, endpoint_velocity_in_base_frame, joint_angles_array):

        ###
        ### Compute the Jacobian inverse at the current set of joint angles
        ###
        # Create a KDL array of the joint angles
        joint_angles_kdl_array = self.convert_joint_angles_array_to_kdl_array(joint_angles_array)
        # Compute the Jacobian at the current joint angles
        jacobian = self.compute_jacobian(joint_angles_kdl_array)
        # Compute the pseudo-inverse
        jacobian_inverse = np.linalg.pinv(jacobian)
        ###
        ### Then, use the Jacobian inverse to compute the required joint velocities
        ###
        # Multiply the Jacobian inverse by the Cartesian velocity
        joint_velocities = jacobian_inverse * np.concatenate([endpoint_velocity_in_base_frame[:3,-1].reshape(3,1),
                                                              T.mat2euler(endpoint_velocity_in_base_frame[:3,:3],axes = 'sxyz').reshape(3,1)])
        # Return the velocities
        return joint_velocities


    # Function to convert from a NumPy array of joint angles, to a KDL array of joint angles (its the same data, just a different type of container)
    def convert_joint_angles_array_to_kdl_array(self, joint_angles_array):
        num_joints = len(joint_angles_array)
        kdl_array = PyKDL.JntArray(num_joints)
        for i in range(num_joints):
            kdl_array[i] = joint_angles_array[i]
        return kdl_array

    # Function to compute the jacobian directly with a numpy array
    def get_jacobian(self, joint_angles_array):
        joint_angles_kdl_array = self.convert_joint_angles_array_to_kdl_array(joint_angles_array)
        # Compute the Jacobian at the current joint angles
        jacobian = self.compute_jacobian(joint_angles_kdl_array)
        return jacobian

    # Function to compute the Jacobian at a particular set of joint angles
    def compute_jacobian(self, joint_angles_kdl_array):
        jacobian = PyKDL.Jacobian(self.num_joints)
        self.jacobian_solver.JntToJac(joint_angles_kdl_array, jacobian)
        jacobian_matrix = self.kdl_array_to_numpy_mat(jacobian)
        return jacobian_matrix

    # Function to get the endpoint pose in the base frame. It returns a 4x4 NumPy matrix homogeneous transformation
    def get_endpoint_pose_matrix(self, joint_angles_array):
        # Get the joint angles
        joint_angles_kdl_array = self.convert_joint_angles_array_to_kdl_array(joint_angles_array)
        # Do the forward kinematics
        endpoint_frame = PyKDL.Frame()
        self.forward_kinematics_solver.JntToCart(joint_angles_kdl_array, endpoint_frame)
        endpoint_frame = self.kdl_frame_to_numpy_mat(endpoint_frame)
        return endpoint_frame

    # Function to convert a KDL array to a NumPy matrix (its the same data, just a different type of container)
    @staticmethod
    def kdl_array_to_numpy_mat(kdl_array):
        mat = np.mat(np.zeros((kdl_array.rows(), kdl_array.columns())))
        for i in range(kdl_array.rows()):
            for j in range(kdl_array.columns()):
                mat[i, j] = kdl_array[i, j]
        return mat

    # Function to convert a KDL 1D array to a NumPy array (its the same data, just a different type of container)
    @staticmethod
    def kdl_1Darray_to_numpy(kdl_array):
        array = np.zeros((kdl_array.rows()))
        for i in range(kdl_array.rows()):
            array[i]=kdl_array[i]

        return array

    @staticmethod
    # Function to convert a KDL transformation frame to a NumPy matrix. Code taken from KDL pose math library.
    def kdl_frame_to_numpy_mat(kdl_frame):
        mat = np.mat(np.zeros([4, 4]))
        for i in range(3):
            for j in range(3):
                mat[i, j] = kdl_frame.M[i, j]
        for i in range(3):
            mat[i, 3] = kdl_frame.p[i]
        mat[3] = [0, 0, 0, 1]
        return mat


if __name__ =='__main__':
    sawyer = Sawyer()

    print(sawyer)
