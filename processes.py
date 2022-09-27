from multiprocessing import Process
from holodex.utils.files import get_yaml_data
from holodex.components import *
from holodex.viz import Hand2DVisualizer, Hand3DVisualizer, MPImageVisualizer
from holodex.viz.visualizer_3d import OculusLeftHandDirVisualizer
from holodex.camera.realsense_camera import RealSenseRobotStream
from holodex.camera.camera_streamer import RobotCameraStreamer


def notify_process_start(notification_statement):
    print("***************************************************************")
    print("     {}".format(notification_statement))
    print("***************************************************************")

def start_robot_cam_stream(cam_serial_num, robot_cam_num):
    notify_process_start("Starting Robot Camera Stream {} Process".format(robot_cam_num))
    camera = RealSenseRobotStream(cam_serial_num, robot_cam_num)
    camera.stream()

def stream_cam_tcp(cam_num, host, port, cam_rotation_angle):
    notify_process_start("Starting Robot Camera TCP Stream Process")
    camera = RobotCameraStreamer(cam_num, host, port, cam_rotation_angle)
    camera.stream()

def start_mp_detector(options):
    notify_process_start("Starting Mediapipe Detection Process")
    detector = MPHandDetector(options['cam_serial_num'], options['resolution'], options['alpha'])
    detector.stream()

def start_oculus_detector(options):
    notify_process_start("Starting OculusVR Detection Process")
    detector = OculusVRHandDetector(options['host'], options['keypoint_stream_port'])
    detector.stream()

def keypoint_transform(detector_type):
    notify_process_start("Starting Keypoint transformation Process")
    transformer = TransformHandCoords(detector_type)
    transformer.stream()

def plot_2d(detector_type, *args):
    notify_process_start("Starting 2D Hand Plotting Process")
    plotter = Hand2DVisualizer(detector_type, *args)
    plotter.stream()

def plot_3d(detector_type):
    notify_process_start("Starting 3D Hand Plotting Process")
    plotter = Hand3DVisualizer(detector_type)
    plotter.stream()

def plot_oculus_left_hand():
    notify_process_start("Starting Oculus Left Hand Direction Plotting Process")
    plotter = OculusLeftHandDirVisualizer()
    plotter.stream()

def viz_hand_stream(rotation_angle):
    notify_process_start("Starting Mediapipe Hand Prediction Image Stream Process")
    visualizer = MPImageVisualizer(rotation_angle)
    visualizer.stream()

def mp_teleop(detector_config):
    notify_process_start("Starting Teleoperation Process")
    teleop = MPDexArmTeleOp()
    teleop.move(detector_config['finger_configs'])

def vr_teleop(detector_config):
    notify_process_start("Starting Teleoperation Process")
    teleop = VRDexArmTeleOp()
    teleop.move(detector_config['finger_configs'])

def deploy_model(configs):
    if configs.model not in ['VINN', 'BC']:
        raise NotImplementedError("{} not implemented".format(configs.model))

    notify_process_start("Starting Deployment Process")
    deployer = DexArmDeploy(configs)
    deployer.solve()

def get_camera_stream_processes(configs):
    robot_camera_processes = []
    robot_camera_stream_processes = []

    for idx, cam_serial_num in enumerate(configs['robot_cam_serial_numbers']):
        robot_camera_processes.append(
            Process(target = start_robot_cam_stream, args = (cam_serial_num, idx + 1, ))
        )

    if 'tracker' in configs.keys(): # Since we use this for deployment as well
        if configs.tracker.type == 'VR':
            if configs.tracker['stream_robot_cam']:
                robot_camera_stream_processes.append(
                    Process(target = stream_cam_tcp, args = (
                        configs.tracker['stream_camera_num'], 
                        configs.tracker['host'], 
                        configs.tracker['robot_cam_stream_port'],
                        configs.tracker['stream_camera_rotation_angle'],
                    ))
                )

    return robot_camera_processes, robot_camera_stream_processes

def get_detector_processes(teleop_configs):
    if teleop_configs.tracker.type == 'MP':
        detection_process = Process(target = start_mp_detector, args = (teleop_configs.tracker, ))
        keypoint_transform_processes = [Process(target = keypoint_transform, args = ('MP', ))]
        
        plotter_processes = []
        if teleop_configs.tracker['visualize_graphs']:
            plotter_processes.append(Process(target = plot_2d, args = (teleop_configs.tracker.type, )))
            plotter_processes.append(Process(target = plot_3d, args = (teleop_configs.tracker.type, ))),
            
        if teleop_configs.tracker['visualize_pred_stream']:
            plotter_processes.append(Process(target = viz_hand_stream, args = (teleop_configs.tracker['pred_stream_rotation_angle'], )))

    elif teleop_configs.tracker.type == 'VR':
        detection_process = Process(target = start_oculus_detector, args = (teleop_configs.tracker, ))
        keypoint_transform_processes = [
            Process(target = keypoint_transform, args = ('VR_RIGHT', )),
            Process(target = keypoint_transform, args = ('VR_LEFT', ))
        ]

        plotter_processes = []
        if teleop_configs.tracker['visualize_right_graphs']:
            plotter_processes.append(Process(target = plot_2d, args = ('VR_RIGHT', teleop_configs.tracker['host'], teleop_configs.tracker['plot_stream_port'], )))
            plotter_processes.append(Process(target = plot_3d, args = ('VR_RIGHT', )))
            
        if teleop_configs.tracker['visualize_left_graphs']:
            plotter_processes.append(Process(target = plot_oculus_left_hand))

    else:
        raise NotImplementedError("No such detector exists!")

    return detection_process, keypoint_transform_processes, plotter_processes

def get_teleop_process(teleop_configs):
    if teleop_configs.tracker.type == 'MP':
        teleop_process = Process(target = mp_teleop, args = (teleop_configs, ))
    elif teleop_configs.tracker.type == 'VR':
        teleop_process = Process(target = vr_teleop, args = (teleop_configs, ))

    return teleop_process

def get_deploy_process(configs):
    deploy_process = Process(target = deploy_model, args = (configs, ))
    return deploy_process