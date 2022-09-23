import sys, os
import cv2
import argparse
from visualOdometry import VO
from helpers import KITTICameraMatrixFromTXT

# TODO: Refactor main to work with videos and KITTI
# TODO: Refactor visualOdometry and add inplace visualization and display trajectory with image frame in same image
# TODO: Refactor featureExtractor

# TODO: Image calibration and Camera Matrix computation
# TODO: 3D point mapping and storage
# TODO: Loop Closing
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Monocular Visual Odometry on Video file / Image directorys")
    # parser.add_argument("file_path", help="The path to the video file that you want to process or the path to the KITTI image folder")
    # parser.add_argument("--kitti", )
    image_folder = "videos/KITTI_SEQ/00/image_2"
    camera_matrix_file = "videos/KITTI_SEQ/00/calib.txt"
    kitti_pose_path = "videos/KITTI_SEQ/poses/00.txt"
    camera_matrix = KITTICameraMatrixFromTXT(camera_matrix_file, 0)
    images = sorted(os.listdir("videos/KITTI_SEQ/00/image_2"))
    vo = VO(camera_matrix, kitti_pose_path=kitti_pose_path)

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        vo.processFrame(img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


# if __name__ == '__main__':
#     cap = cv2.VideoCapture("videos/test1.mp4")
#     vo = VO()
#     if (cap.isOpened() == False):
#         print("Error opening video stream or file")
#     while (cap.isOpened()):
#         ret, frame = cap.read()
#         if ret == True:
#             vo.processFrame(frame, display=True)
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

