import time
import hydra
from processes import get_camera_stream_processes, get_deploy_process

@hydra.main(version_base = '1.2', config_path='configs', config_name='deploy')
def main(configs):
    robot_camera_processes, _ = get_camera_stream_processes(configs)
    deploy_process = get_deploy_process(configs)

    # Starting all the processes
    # Camera processes
    for process in robot_camera_processes:
        process.start()
        time.sleep(2)

    # Deploy process
    deploy_process.start()

    # Joining all the processes
    for process in robot_camera_processes:
        process.join()

    deploy_process.join()

if __name__ == '__main__':
    main()