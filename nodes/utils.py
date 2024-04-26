import numpy as np
from scipy.spatial.transform import Rotation as Rot

import rospy
import ros_numpy as rnp

from segment_track.observation import Observation
import segment_slam_msgs.msg as segment_slam_msgs
import geometry_msgs.msg as geometry_msgs

def observation_from_msg(observation_msg: segment_slam_msgs.Observation):
    """
    Convert observation message to observation data class

    Args:
        observation_msg (segment_slam_msgs.Observation): observation message

    Returns:
        Observation: observation data class
    """
    observation = Observation(
        time=observation_msg.stamp.to_sec(),
        pose=rnp.numpify(observation_msg.pose),
        pixel=np.array(observation_msg.pixel),
        width=observation_msg.mask_width,
        height=observation_msg.mask_height,
        mask=np.array(observation_msg.mask).reshape(
            (observation_msg.img_height, observation_msg.img_width)
        ) if observation_msg.mask else None,
        mask_downsampled=np.array(observation_msg.mask).reshape(
            (observation_msg.img_height, observation_msg.img_width)
        ) if observation_msg.mask else None,
        point_cloud=(np.array(observation_msg.point_cloud).reshape((-1, 3)) 
                     if observation_msg.point_cloud else None),
    )
    return observation

def observation_to_msg(observation: Observation):
# def observation_to_msg(observation: Observation, img_width: int, img_height: int):
    """
    Convert observation data class to observation message

    Args:
        observation (Observation): observation data class

    Returns:
        segment_slam_msgs.Observation: observation message
    """
    observation_msg = segment_slam_msgs.Observation(
        stamp=rospy.Time.from_sec(observation.time),
        pose=geometry_msgs.Pose(
            position=rnp.msgify(geometry_msgs.Point, observation.pose[:3,3]),
            orientation=rnp.msgify(geometry_msgs.Quaternion, Rot.from_matrix(observation.pose[:3,:3]).as_quat())
        ),
        pixel=observation.pixel.tolist(),
        img_width=int(observation.mask_downsampled.shape[1]),
        img_height=int(observation.mask_downsampled.shape[0]),
        mask_width=int(observation.width),
        mask_height=int(observation.height),
        mask=observation.mask_downsampled.flatten().astype(np.int8).tolist() if observation.mask is not None else None,
        point_cloud=(observation.point_cloud.flatten().tolist() 
                     if observation.point_cloud is not None else None),
    )
    return observation_msg