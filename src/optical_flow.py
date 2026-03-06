import cv2

# Compute dense optical flow between two frames using Farneback's method.
# Returns motion vectors (dx,dy) for each pixel indicating how the image. moved from the previous frame to the current frame.
# the format is: cv2.calcOpticalFlowFarneback( prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags )
def compute_flow(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(prev, curr,None, 0.5,3,15,3, 5,1.2, 0)
    return flow