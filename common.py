import cv2
import numpy as np

def moyenne_image(video, nbr):
    
    """
    cette fonction calcule la moyenne d'un certain nbr d'image 
    """
    cap=cv2.VideoCapture(video)
    tab_image=[]
    for f in range(nbr):
        ret, frame=cap.read()
        # verification de la lecture est bien fait
        if ret is False:
            break
        image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tab_image.append(image)
    tab_image=np.array(tab_image)
    # libérer la ressource
    cap.release()
    # return le calcule de la moyenne d'image
    return np.mean(tab_image, axis=0)

def calcul_mask(image, fond, seuil):
    
    """
    calcule le mask a partire de l'image capture et de fond selon un certain seuil (reglable en fonction des condition de l'image)
        image : c'est la frame capture lors de la video capture
        fond : c'est l'image de fond calculer par moyenn_image()
        seuil : c'est la valeur qu'on determine selon la luminosite de l'image 
    
    """
    # change the color image to gray
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width=image.shape
    # initialisation de mask
    mask=np.zeros([height, width], np.uint8)
    # conversion de l'image a une matrice de type int
    image=image.astype(np.int32)
    for y in range(height):
        for x in range(width):
            if abs(fond[y][x]-image[y][x])>seuil:
                mask[y][x]=255
    # etape de nettoyage de mask :
    # determiner la taille de kernel
    kernel=np.ones((5, 5), np.uint8)
    # application de l'erosion (le OU logique )
    mask=cv2.erode(mask, kernel, iterations=1)
    # application de la dilatation (le ET logique )
    mask=cv2.dilate(mask, kernel, iterations=3)
    return mask
