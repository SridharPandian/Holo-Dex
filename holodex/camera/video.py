import os
import cv2

from holodex.utils.files import make_dir
from holodex.utils.network import frequency_timer, ImageSubscriber
from holodex.constants import *

class StreamVideoRecorder(object):
    def __init__(self, robot_cam_num, storage_path):
        self.storage_path = storage_path
        make_dir(os.path.dirname(storage_path))

        self.robot_image_subscriber = ImageSubscriber(
            '/robot_camera_{}/color_image'.format(robot_cam_num),
            'robot_camera_{}_recorder'.format(robot_cam_num)
        )

        self.frequency_timer = frequency_timer(RECORD_FPS)

    def generate_video(self):
        print('Creating the video writer!')
        video_writer = cv2.VideoWriter(self.storage_path, cv2.VideoWriter_fourcc('M','J','P','G'), RECORD_FPS, (1280, 720))

        print('Recording the video...')
        while True:
            if self.robot_image_subscriber.get_image() is  None:
                continue
            
            try:
                video_writer.write(self.robot_image_subscriber.get_image())
                self.frequency_timer.sleep()

            except KeyboardInterrupt:
                break

        print('Storing Video!')
        video_writer.release()


class FolderVideoRecorder(object):
    def __init__(self, image_frames_path, video_storage_path):
        self.image_names = os.listdir(image_frames_path)
        self.image_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.image_paths = [os.path.join(image_frames_path, image_name) for image_name in self.image_names]
        self.video_storage_path = video_storage_path

    def generate_video(self, image_dims, fps):
        video_writer = cv2.VideoWriter(self.video_storage_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (image_dims[0], image_dims[1]))

        for image_path in self.image_paths:
            image = cv2.imread(image_path)
            cv2.imshow('',image)
            video_writer.write(image)

        video_writer.release()