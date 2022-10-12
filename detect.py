import time
import hydra
from processes import get_detector_processes

@hydra.main(version_base = '1.2', config_path='configs', config_name='teleop')
def main(configs):
    detection_process, keypoint_transform_processes, plotter_processes = get_detector_processes(configs)

    # Starting all the processes
    detection_process.start()
    time.sleep(2)

    for process in keypoint_transform_processes:
        process.start()

    for process in plotter_processes:
        process.start()

    # Joining all the processes
    detection_process.join()

    for process in keypoint_transform_processes:
        process.join()

    for process in plotter_processes:
        process.join()

if __name__ == '__main__':
    main()