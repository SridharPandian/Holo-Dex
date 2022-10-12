import cv2
import numpy as np
import rospy
import pyrealsense2 as rs
from holodex.utils.network import *
from holodex.constants import *
from mediapipe import solutions as mp_solutions

class MPHandDetector(object):
    def __init__(self, cam_serial_num, resolution, alpha):
        # Initializing the ROS node
        rospy.init_node("mediapipe_keypoint_extractor")

        # Disabling scientific notations
        np.set_printoptions(suppress=True)

        # Storing options
        self.alpha = alpha
        self.resolution = resolution

        # Medaipipe Initializations
        self._mediapipe_hands = mp_solutions.hands
        self._mediapipe_drawing = mp_solutions.drawing_utils
        
        # Initializing Realsense pipeline
        self._start_realsense(cam_serial_num, self.resolution[0], self.resolution[1], MP_FREQ, MP_PROCESSING_PRESET)

        # Creating an empty keypoint array
        self.keypoints = np.empty((MP_NUM_KEYPOINTS, 3))

        # Initializing ROS publishers
        self.keypoint_publisher = FloatArrayPublisher(publisher_name = MP_KEYPOINT_TOPIC)
        self.pred_boolean_publisher = BoolPublisher(publisher_name = MP_PRED_BOOL_TOPIC)
        self.og_rgb_image_publisher = ImagePublisher(publisher_name = MP_RGB_IMAGE_TOPIC, color_image = True)
        self.og_depth_image_publisher = ImagePublisher(publisher_name = MP_DEPTH_IMAGE_TOPIC, color_image = False)
        self.pred_rgb_image_publisher = ImagePublisher(publisher_name = MP_PRED_RGB_IMAGE_TOPIC, color_image = True)
        self.pred_depth_image_publisher = ImagePublisher(publisher_name = MP_PRED_DEPTH_IMAGE_TOPIC, color_image = False)

        # Initializing CvBridge
        self.bridge = CvBridge()

    def _start_realsense(self, cam_serial_num, width, height, fps, processing_preset):
        config = rs.config()
        pipeline = rs.pipeline()
        config.enable_device(cam_serial_num)

        # Enabling camera streams
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Starting the pipeline
        cfg = pipeline.start(config)
        device = cfg.get_device()

        # Setting the depth mode to high accuracy mode
        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, processing_preset) 
        self.realsense = pipeline

        # Obtaining the color intrinsics matrix for aligning the color and depth images
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy], 
            [0, 0, 1]
        ])

        # Align function - aligns other frames with the color frame
        self.align = rs.align(rs.stream.color)
        print("Started the Realsense pipeline!")

    def get_rgb_depth_images(self):
        frames = None

        while frames is None:
            # Obtaining and aligning the frames
            frames = self.realsense.wait_for_frames()
            aligned_frames = self.align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            colorized_depth_frame = rs.colorizer().colorize(aligned_depth_frame)
            color_frame = aligned_frames.get_color_frame()

            # Getting the images from the frames
            colored_depth_image = np.asanyarray(colorized_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        return color_image, colored_depth_image, aligned_depth_frame

    def _get_depth_value(self, x_pixel, y_pixel, depth_frame):
        pixel_range = [-2, -1, 0, 1, 2]

        depth_values = []

        for x_range in pixel_range:
            for y_range in pixel_range:
                
                curr_x_pixel = x_pixel + x_range
                curr_y_pixel = y_pixel + y_range

                if curr_x_pixel >= 0 and curr_x_pixel < self.resolution[0] and curr_y_pixel >= 0 and curr_y_pixel < self.resolution[1]:
                    depth = depth_frame.get_distance(curr_x_pixel, curr_y_pixel)

                    if depth != 0:
                        depth_values.append(depth)

        if len(depth_values) > 0:
            return np.average(depth_values)
        else:
            return 0

    def _temporal_depth_average(self, idx, new_depth_value, alpha):
        if self.keypoints[idx][2] != 0:
            return alpha * new_depth_value + (1 - alpha) * self.keypoints[idx][2]
        else:
            return new_depth_value

    def get_keypoints(self, hand_landmarks, depth_frame):
        wrist_coords = np.empty(3)

        knuckle_coords = np.empty((5, 3))
        knuckle_counter = 0
        
        tip_coords = np.empty((5, 3))
        tip_counter = 0
        
        for idx, point in enumerate(hand_landmarks.landmark):
            # Ignoring points which are out of range
            if point.x < 0 or point.y < 0 or point.x > 1 or point.y > 1: 
                return None

            if idx == 0: # For the wrist
                # Getting the pixel value
                x_pixel, y_pixel = int(self.resolution[0] * point.x), int(self.resolution[1] * point.y)

                # Obtaining the depth value for the wrist joint
                new_depth = self._get_depth_value(x_pixel, y_pixel, depth_frame)

                # # Using the temporal moving average filter
                if new_depth > 0:
                    wrist_coords[2] = self._temporal_depth_average(0, new_depth, alpha = self.alpha)

                # Finding the x and y coordinates
                wrist_coords[0] = wrist_coords[2] * (x_pixel - self.intrinsics_matrix[0][2]) / self.intrinsics_matrix[0][0]
                wrist_coords[1] = wrist_coords[2] * (y_pixel - self.intrinsics_matrix[1][2]) / self.intrinsics_matrix[1][1]

            if idx in MP_OG_KNUCKLES: # For the knuckles
                # Getting the pixel value
                x_pixel, y_pixel = int(self.resolution[0] * point.x), int(self.resolution[1] * point.y)

                # Setting the knuckle point depths as the wrist joint's depth value
                new_depth = wrist_coords[2] 
                
                # Using the temporal moving average filter
                if new_depth > 0:
                    knuckle_coords[knuckle_counter][2] = self._temporal_depth_average(knuckle_counter + 1, new_depth, alpha = self.alpha)

                # Finding the x and y coordinates
                knuckle_coords[knuckle_counter][0] = knuckle_coords[knuckle_counter][2] * (x_pixel - self.intrinsics_matrix[0][2]) / self.intrinsics_matrix[0][0]
                knuckle_coords[knuckle_counter][1] = knuckle_coords[knuckle_counter][2] * (y_pixel - self.intrinsics_matrix[1][2]) / self.intrinsics_matrix[1][1]

                knuckle_counter += 1

            if idx in MP_OG_FINGERTIPS: # For the tips
                # Getting the pixel value
                x_pixel, y_pixel = int(self.resolution[0] * point.x), int(self.resolution[1] * point.y)

                # Setting the tip point depths as the wrist joint's depth value
                new_depth = self._get_depth_value(x_pixel, y_pixel, depth_frame) 
                
                # Using the temporal moving average filter
                if new_depth > 0:
                    tip_coords[tip_counter][2] = self._temporal_depth_average(tip_counter + 6, new_depth, alpha = self.alpha)

                # Finding the x and y coordinates
                tip_coords[tip_counter][0] = tip_coords[tip_counter][2] * (x_pixel - self.intrinsics_matrix[0][2]) / self.intrinsics_matrix[0][0]
                tip_coords[tip_counter][1] = tip_coords[tip_counter][2] * (y_pixel - self.intrinsics_matrix[1][2]) / self.intrinsics_matrix[1][1]

                tip_counter += 1

        return np.vstack([wrist_coords, knuckle_coords, tip_coords])

    def stream(self):
        with self._mediapipe_hands.Hands(
            static_image_mode = False,
            max_num_hands = MAX_NUM_HANDS, # Limiting the number of hands detected in the image to 1
            min_detection_confidence = PRED_CONFIDENCE,
            min_tracking_confidence = PRED_CONFIDENCE) as hand:
            # Starting the video loop
            while True:
                # Obtaining the color and depth images
                color_image, depth_image, depth_frame = self.get_rgb_depth_images()

                # Publishing the original color and depth images
                self.og_rgb_image_publisher.publish(color_image)
                self.og_depth_image_publisher.publish(depth_image)

                # Preprocessing the color image and predicting the hand pose
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                color_image.flags.writeable = False
                prediction = hand.process(color_image)

                # Post-processing the color image
                color_image.flags.writeable = True
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                predicted = False

                # Adding detections in the image
                if prediction.multi_hand_landmarks: 
                    # Obtaining the keypoints from the prediction
                    obtained_keypoints = self.get_keypoints(prediction.multi_hand_landmarks[0], depth_frame)

                    if obtained_keypoints is None:
                        continue

                    # Assigning the keypoints
                    self.keypoints = obtained_keypoints
                    predicted = True

                    # Drawing the points in the color image
                    self._mediapipe_drawing.draw_landmarks(
                        color_image,
                        prediction.multi_hand_landmarks[0],
                        self._mediapipe_hands.HAND_CONNECTIONS
                    )
                    
                    # Drawing the points in the depth image
                    self._mediapipe_drawing.draw_landmarks(
                        depth_image,
                        prediction.multi_hand_landmarks[0],
                        self._mediapipe_hands.HAND_CONNECTIONS
                    )

                    # Publishing the predicted data
                    self.pred_rgb_image_publisher.publish(color_image)
                    self.pred_depth_image_publisher.publish(depth_image)
                    self.keypoint_publisher.publish(self.keypoints.flatten().tolist())  

                self.pred_boolean_publisher.publish(predicted)