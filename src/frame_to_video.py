import cv2
import glob
import os


def frames_to_video(fps, save_path, frames_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, (640, 480))
    imgs = glob.glob(frames_path + "/*.jpg")
    for i in imgs:
        frame = cv2.imread(i)
        videoWriter.write(frame)
    videoWriter.release()
    return


if __name__ == '__main__':
    frames_to_video(15, "../videos/MOT16-06.mp4", 'E:/PyProjects/datasets/MOT16/test/MOT16-06/img1')
