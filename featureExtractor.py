import cv2

# TODO: Add Grid Extraction
class FeatureExtractor:
    nFeatures = 3000
    orb = None
    grey = False
    brute_force = True
    filter_matches = False
    filter_threshold = 0.75

    def __init__(self, n_features=3000, grey=False, brute_force=True, filter_matches=False, filter_threshold=0.75):
        self.nFeatures = n_features
        self.orb = cv2.ORB_create(self.nFeatures)
        self.grey = grey
        self.brute_force = brute_force
        self.filter_matches = filter_matches
        self.filter_threshold = filter_threshold

    def findKeypoints(self, img):
        if self.grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.orb.detect(img, None)

    # For increased performance just compute keypoints and descriptors once and pass them to the matcher
    def computeDescriptors(self, img, keypoints=None):
        if self.grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if keypoints is None:
            return self.orb.compute(img, self.findKeypoints(img))
        else:
            return self.orb.compute(img, keypoints)

    def __match(self, des1, des2):
        # TODO: implement FLANN matcher and Condition
        matches = None
        if self.filter_matches:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.knnMatch(des1, des2, k=2)
            matches = sorted(matches, key=lambda x: x.distance)
            good = []
            for m, n in matches:
                # If distances are within a certain threshold, add to good matches
                if m.distance < self.filter_threshold * n.distance:
                    good.append([m])
            return good
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches

    def matchKeypointsFromDescriptors(self, des1, des2):
        return self.__match(des1, des2)

    def matchKeypointsFromImages(self, img1, img2):
        kp1, des1 = self.computeDescriptors(img1)
        kp2, des2 = self.computeDescriptors(img2)
        return self.matchKeypointsFromDescriptors(des1, des2)

    #TODO: Outsorce drawing to visualOdometry.py and add inframe drawing
    def drawKeypointsOnImage(self, img):
        return cv2.drawKeypoints(img, self.findKeypoints(img, self.grey), None, color=(0, 255, 0), flags=0)
