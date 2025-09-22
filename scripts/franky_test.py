from franky import *
ROBOT_IP = "192.168.1.1"

print("Init franka robot...")
fr3_robot = Robot(ROBOT_IP)
fr3_gripper = Gripper(ROBOT_IP)

state = fr3_robot.state
 
# Get the robot's cartesian state
cartesian_state = fr3_robot.current_cartesian_state
robot_pose = cartesian_state.pose  # Contains end-effector pose and elbow position
ee_pose = robot_pose.end_effector_pose
print("ee_pose: ", ee_pose)
 
# Get the robot's joint state
joint_state = fr3_robot.current_joint_state
joint_pos = joint_state.position
print("join pos: ", joint_pos)