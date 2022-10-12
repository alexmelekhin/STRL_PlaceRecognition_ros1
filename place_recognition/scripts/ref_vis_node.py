#!/usr/bin/env python3

import os
from time import time

import yaml
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
import message_filters

from std_msgs.msg import String
from sensor_msgs.msg import Image

from src.utils import convert_dict_to_tuple


class RefVisNode:

    def __init__(self):
        rospy.init_node('ref_vis_node', log_level=rospy.DEBUG)

        config_path = rospy.get_param('~config_path')

        with open(config_path) as f:
            data = yaml.safe_load(f)

        self.config = convert_dict_to_tuple(data)

        self.sub_ref_ts = message_filters.Subscriber('/visual_pr/ref_image_ts', String, queue_size=1)
        self.pub_ref_image = rospy.Publisher('/visual_pr/ref_image', Image, queue_size=1)

        self.sub_ref_ts.registerCallback(self.on_ref_image_ts)

        self.bridge = CvBridge()
    
        rospy.loginfo("ref_vis_node is ready")

    def on_ref_image_ts(self, ref_image_ts_msg: String):
        rospy.logdebug("Received String message")
        start_time = time()
        ref_image_ts = ref_image_ts_msg.data
        ref_image_path = os.path.join(self.config.database.images.dir, ref_image_ts+'.png')

        ref_image = cv2.imread(ref_image_path)

        try:
            ref_image_msg = self.bridge.cv2_to_imgmsg(ref_image, "passthrough")
        except CvBridgeError as e:
            rospy.logerror(e)

        self.pub_ref_image.publish(ref_image_msg)

        time_taken = time() - start_time
        rospy.logdebug(f"Processed the message and published the image. Time taken: {time_taken*1000:.3f} ms")


def main(args=None):
    visual_pr_node = RefVisNode()
    rospy.spin()


if __name__ == '__main__':
    main()
