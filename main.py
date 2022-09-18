import cv2
from visualOdometry import VO

# TODO: Outsorce frame processing to a seperate class
# TODO: Add inframe matching
# TODO: Add k-best matching and remove inliers
# TODO: Compute EssentialMatrixTransform and RT from EssentialMatrix
# TODO: Compute Trajectory from RT

# TODO: Image calibration and Camera Matrix computation
# TODO: Loop Closing

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cap = cv2.VideoCapture("videos/test1.mp4")
    vo = VO()
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            vo.processFrame(frame, display=True)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
