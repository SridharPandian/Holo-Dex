import time

import hydra
from processes import get_camera_stream_processes

@hydra.main(version_base = '1.2', config_path='configs', config_name='teleop')
def main(configs):    
    # Obtaining all the robot streams
    robot_camera_processes, _ = get_camera_stream_processes(configs)

    # Starting all the processes
    # Camera processes
    for process in robot_camera_processes:
        process.start()
        time.sleep(2)

    # Joining all the processes
    for process in robot_camera_processes:
        process.join()

if __name__ == '__main__':
    main()