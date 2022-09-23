import cv2
import numpy as np
from featureExtractor import FeatureExtractor



class VO:
    img1 = None
    img2 = None
    extractor = None
    cameraMatrix = None
    R = np.identity(3)
    t = np.array([300, 300, 300], dtype=np.float32).reshape(3, 1)
    trajectory = np.zeros((600, 600, 3), dtype=np.uint8)
    id = 0
    pose = None

    def __init__(self, n_features=3000, grey=False, brute_force=True, filter_matches=True, filter_threshold=0.8, camera_matrix = None):
        self.extractor = FeatureExtractor(n_features, grey, brute_force, filter_matches, filter_threshold)
        self.cameraMatrix = camera_matrix
        with open("videos/KITTI_SEQ/poses/00.txt") as f:
            self.pose = f.readlines()

    def get_absolute_scale(self):
        pose = self.pose[self.id - 1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.pose[self.id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])

        true_vect = np.array([[x], [y], [z]])
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])

        return np.linalg.norm(true_vect - prev_vect)

    def processFrame(self, img, display):
        self.id += 1
        if self.img1 is None:
            self.img1 = img
            return
        else:
            self.img2 = img
            key1, des1 = self.extractor.computeDescriptors(self.img1)
            key2, des2 = self.extractor.computeDescriptors(self.img2)
            pt1, pt2, matches = self.extractor.matchKeypointsFromDescriptors(key1, des1, key2, des2)


            #TODO: Check whether camera matrix is none
            E, mask = cv2.findEssentialMat(pt1, pt2, self.cameraMatrix[:, :3], cv2.RANSAC, 0.999, 1.0, None)
            #TODO: Input mask and camera matrix
            #TODO Update R and t
            z, R_curr, t_curr, mask = cv2.recoverPose(E, pt1, pt2, self.cameraMatrix[:, :3])
            print(sum(mask))

            self.t = self.t + self.get_absolute_scale() * np.dot(self.R, t_curr)
            self.R = np.dot(self.R, R_curr)

            cv2.circle(self.trajectory, (int(self.t[0, 0]), int(self.t[2, 0])), 1, (255, 0, 0), 2)
            cv2.imshow("Trajectory", self.trajectory)
            self.img1 = self.img2

