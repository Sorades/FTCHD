import cv2

# To avoid CPU overhead bug raised by albumentations
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
