import cv2
import numpy as np
import common

# affichage de fond 

image=common.moyenne_image('autoroute.mp4', 100)
cv2.imshow('fond', image.astype(np.uint8))
cv2.waitKey()
cv2.release()
cv2.destroyAllWindows()
