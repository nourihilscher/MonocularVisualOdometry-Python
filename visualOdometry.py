import cv2
from featureExtractor import FeatureExtractor

class VO:
    img1 = None
    img2 = None
    extractor = None

    def __init__(self, n_features=3000, grey=False, brute_force=True, filter_matches=False, filter_threshold=0.75):
        self.extractor = FeatureExtractor(n_features, grey, brute_force, filter_matches, filter_threshold)

    def processFrame(self, img, display=True):
        if (self.img1 is None):
            self.img1 = img
            return
        else:
            self.img2 = img
            if display:
                key1, des1 = self.extractor.computeDescriptors(self.img1)
                key2, des2 = self.extractor.computeDescriptors(self.img2)
                matches = self.extractor.matchKeypointsFromDescriptors(des1, des2)
                cv2.imshow('MonocularVisualOdometry', cv2.drawMatches(self.img1, key1, self.img2, key2, matches, None, flags=2))
            self.img1 = self.img2




