import hydra
import time
from processes import get_camera_stream_processes, get_detector_processes, get_teleop_process

@hydra.main(version_base = '1.2', config_path='configs', config_name='teleop')
def main(configs):    
    # Obtaining all the robot streams
    robot_camera_processes, robot_camera_stream_processes = get_camera_stream_processes(configs)
    detection_process, keypoint_transform_processes, plotter_processes = get_detector_processes(configs)
    teleop_process = get_teleop_process(configs)

    # Starting all the processes
    # Camera processes
    for process in robot_camera_processes:
        process.start()
        time.sleep(2)

    for process in robot_camera_stream_processes:
        process.start()

    # Detection processes
    detection_process.start()
    time.sleep(2)

    for process in keypoint_transform_processes:
        process.start()

    for process in plotter_processes:
        process.start()
    
    time.sleep(2)

    # Teleop process
    teleop_process.start()

    # Joining all the processes
    for process in robot_camera_processes:
        process.join()

    for process in robot_camera_stream_processes:
        process.join()

    detection_process.join()

    for process in keypoint_transform_processes:
        process.join()

    for process in plotter_processes:
        process.join()

    teleop_process.join()

if __name__ == '__main__':
    main()