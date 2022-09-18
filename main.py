import cv2
import os


class OrbExtractor:
    nFeatures = 3000
    orb = None

    def __init(self):
        self.orb = cv2.ORB_create(self.nFeatures)

    def __init__(self, n_features):
        self.nFeatures = n_features
        self.orb = cv2.ORB_create(self.nFeatures)

    def findKeypoints(self, img, grey=False):
        if grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.orb.detect(img, None)

    def computeKeypointsAndDescriptors(self, img, grey=False):
        if grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.orb.compute(img, self.findKeypoints(img))

    def drawKeypointsOnImage(self, img, grey=False):
        return cv2.drawKeypoints(img, self.findKeypoints(img, grey), None, color=(0, 255, 0), flags=0)

    def matchKeypoints(self, img1, img2, grey=False, brute_force=True):
        kp1, des1 = self.computeKeypointsAndDescriptors(img1, grey)
        kp2, des2 = self.computeKeypointsAndDescriptors(img2, grey)
        matches = None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # TODO: implement FLANN matcher and Condition

        matches = sorted(matches, key=lambda x: x.distance)
        return matches

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TODO: Map orbs from one scene to another
    # TODO: Compute EssentialMatrixTransform
    # TODO: Image calibration and Camera Matrix computation
    cap = cv2.VideoCapture("videos/test1.mp4")
    img1 = None
    img2 = None
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            extractor = OrbExtractor(3000)
            #cv2.imshow('VisualOdometryGrey', extractor.drawKeypointsOnImage(frame, grey=True))
            #cv2.imshow("VisualOdometry", extractor.drawKeypointsOnImage(frame, grey=False))
            if img1 is None:
                img1 = frame
                continue
            img2 = frame
            cv2.imshow('VisualOdometryGrey', cv2.drawMatches(img1, extractor.findKeypoints(img1), img2, extractor.findKeypoints(img2), extractor.matchKeypoints(img1, img2), None, flags=2))
            img1 = img2
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
