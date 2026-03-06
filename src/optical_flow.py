import cv2

def compute_flow(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(prev, curr,None, 0.5,3,15,3, 5,1.2, 0)
    return flow