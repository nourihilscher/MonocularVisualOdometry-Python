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

    def __init__(self, camera_matrix, n_features=3000, grey=False, brute_force=True, filter_matches=True, filter_threshold=0.8, kitti_pose_path=None):
        self.extractor = FeatureExtractor(n_features, grey, brute_force, filter_matches, filter_threshold)
        self.cameraMatrix = camera_matrix
        if kitti_pose_path is not None:
            with open(kitti_pose_path, 'r') as f:
                self.pose = f.readlines()

    # It is not possible to obtain a trajectory estimate without drift when using only one camera.
    # For monocular VO we are sort of cheating here
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

    def computeEssentialMatrix(self, pt1, pt2):
        return cv2.findEssentialMat(pt1, pt2, self.cameraMatrix[:, :3], cv2.RANSAC, 0.999, 1.0)

    def recoverPose(self, E, pt1, pt2, mask=None):
        return cv2.recoverPose(E, pt1, pt2, self.cameraMatrix[:, :3], mask=mask)

    # Inline visualization of matched keypoints
    def visualizeMatches(self, keypoints1, keypoints2):
        img = self.img2.copy()
        for idx in range(0, keypoints1.shape[0]):
            cv2.circle(img, (int(keypoints2[idx, 0]), int(keypoints2[idx, 1])), color=(0, 255, 0), radius=2)
            cv2.line(img, (int(keypoints1[idx, 0]), int(keypoints1[idx, 1])), (int(keypoints2[idx, 0]), int(keypoints2[idx, 1])), (255, 0, 0), 1)
        cv2.imshow("MonocularVisualOdometry", img)

    def processFrame(self, img):
        if self.img1 is None:
            self.id += 1
            self.img1 = img
            return
        else:
            self.img2 = img
            key1, des1 = self.extractor.computeDescriptors(self.img1)
            key2, des2 = self.extractor.computeDescriptors(self.img2)
            pt1, pt2, matches = self.extractor.matchKeypointsFromDescriptors(key1, des1, key2, des2)

            # Ensure we are having enough matches to calculate essential matrix with 8 point algorithm
            if pt1.shape[0] < 8:
                return

            E, mask = self.computeEssentialMatrix(pt1, pt2)
            self.visualizeMatches(pt1[mask.flatten() == 1], pt2[mask.flatten() == 1])
            _, R_curr, t_curr, _ = self.recoverPose(E, pt1, pt2, mask)

            scale = 1
            if self.pose is not None:
                scale = self.get_absolute_scale()
                self.id += 1
            self.t = self.t + scale * np.dot(self.R, t_curr)
            self.R = np.dot(self.R, R_curr)

            cv2.circle(self.trajectory, (int(self.t[0, 0]), int(self.t[2, 0])), 1, (255, 0, 0), 2)
            cv2.imshow("Trajectory", self.trajectory)
            self.img1 = self.img2
