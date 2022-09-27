import os.path as path
import holodex

# Allegro
ALLEGRO_JOINT_STATE_TOPIC = '/allegroHand/joint_states'
ALLEGRO_COMMANDED_JOINT_STATE_TOPIC = '/allegroHand/commanded_joint_states'
ALLEGRO_JOINTS_PER_FINGER = 4
ALLEGRO_JOINT_OFFSETS = {
    'index': 0,
    'middle': 4,
    'ring': 8,
    'thumb': 12
}
ALLEGRO_HOME_POSITION = [ 0., -0.17453293, 0.78539816, 0.78539816, 0., -0.17453293, 0.78539816, 0.78539816, 0.08726646, -0.08726646, 0.87266463, 0.78539816, 1.04719755, 0.43633231, 0.26179939, 0.78539816]

# Kinova
KINOVA_JOINT_STATE_TOPIC = '/j2n6s300_driver/out/joint_state'

# Used for tasks
KINOVA_POSITIONS = {
    'flat': [-0.39506303664033293, 3.5573131650155982, 0.6726404554757113, 3.6574022156318287, 1.7644077385694936, 3.971040566681588],
    'slide': [-1.4591583449534833, 3.719499409085315, 1.4473843766161887, 5.551245841607734, 2.172958354550533, 0.9028881088391743],
    'opening': {
        'one_day': [5.670515511026934, 3.9037270396009633, 0.9027256560126795, 3.5419925318965904, 1.860775021562338, 5.610006360531461],
        'gatorade': [5.64553506000199, 3.57880434238032, 0.9980285400968463, 3.727822852756248, 1.53733397401416, 5.375678544640914],
        'blender': [5.651544749317862, 3.694911508005049, 0.9472571715482214, 3.6556964609536324, 1.6506910263392192, 5.4651170814257],
        'koia': [5.656182378040127, 3.7159848353947376, 0.9378061454717482, 3.6418381695891875, 1.6748827822073353, 5.486710524176281],
        'sprite': []
    }

}

# Calibration file paths
CALIBRATION_FILES_PATH = path.join(path.dirname(holodex.__path__[0]), 'calibration_files')

# Realsense Camera parameters
NUM_CAMS = 3
CAM_FPS = 30
WIDTH = 1280
HEIGHT = 720
PROCESSING_PRESET = 1 # High accuracy post-processing mode
VISUAL_RESCALE_FACTOR = 2
RECORD_FPS = 5

# Mediapipe detector
# ROS Topics
MP_RGB_IMAGE_TOPIC = "/mediapipe/original/color_image"
MP_DEPTH_IMAGE_TOPIC = "/mediapipe/original/depth_image"

MP_KEYPOINT_TOPIC = "/mediapipe/predicted/keypoints"
MP_PRED_BOOL_TOPIC = "/mediapipe/predicted/detected_boolean"

MP_PRED_RGB_IMAGE_TOPIC = "/mediapipe/predicted/color_image"
MP_PRED_DEPTH_IMAGE_TOPIC = "/mediapipe/predicted/depth_image"

MP_HAND_TRANSFORM_COORDS_TOPIC = "/mediapipe/predicted/transformed_keypoints"

# File paths
MP_THUMB_BOUNDS_PATH = path.join(CALIBRATION_FILES_PATH, 'mp_bounds.npy')

# Joint information
MP_NUM_KEYPOINTS = 11
MP_THUMB_BOUND_VERTICES = 4

MP_OG_KNUCKLES = [1, 5, 9, 13, 17]
MP_OG_FINGERTIPS = [4, 8, 12, 16, 20]

MP_JOINTS = {
    'metacarpals': [1, 2, 3, 4, 5],
    'knuckles': [2, 3, 4, 5],
    'thumb': [1, 6],
    'index':[2, 7],
    'middle': [3, 8],
    'ring': [4, 9],
    'pinky': [5, 10] 
}


MP_VIEW_LIMITS = {
    'x_limits': [-0.12, 0.12],
    'y_limits': [-0.02, 0.2],
    'z_limits': [0, 0.06]
}

# Other params
MP_PROCESSING_PRESET = 2 # Hands post-processing mode - Hands mode
PRED_CONFIDENCE = 0.95
MAX_NUM_HANDS = 1
MP_FREQ = 30


# VR detector 
# ROS Topic names
VR_RIGHT_HAND_KEYPOINTS_TOPIC = '/OculusVR/right_hand_keypoints'
VR_LEFT_HAND_KEYPOINTS_TOPIC = '/OculusVR/left_hand_keypoints'

VR_RIGHT_TRANSFORM_COORDS_TOPIC = "/OculusVR/transformed_right"
VR_LEFT_TRANSFORM_DIR_TOPIC = "/OculusVR/left_dir_vectors"

# File paths
VR_THUMB_BOUNDS_PATH = path.join(CALIBRATION_FILES_PATH, 'vr_thumb_bounds.npy')
VR_DISPLAY_THUMB_BOUNDS_PATH = path.join(CALIBRATION_FILES_PATH, 'vr_thumb_plot_bounds.npy')
VR_2D_PLOT_SAVE_PATH = path.join(CALIBRATION_FILES_PATH, 'oculus_hand_2d_plot.jpg')

# Joint Information
OCULUS_NUM_KEYPOINTS = 24
VR_THUMB_BOUND_VERTICES = 8

OCULUS_JOINTS = {
    'metacarpals': [2, 6, 9, 12, 15],
    'knuckles': [6, 9, 12, 16],
    'thumb': [2, 3, 4, 5, 19],
    'index': [6, 7, 8, 20],
    'middle': [9, 10, 11, 21],
    'ring': [12, 13, 14, 22],
    'pinky': [15, 16, 17, 18, 23]
}

OCULUS_VIEW_LIMITS = {
    'x_limits': [-0.04, 0.04],
    'y_limits': [-0.02, 0.25],
    'z_limits': [-0.04, 0.04]
}

# Other params
VR_FREQ = 60