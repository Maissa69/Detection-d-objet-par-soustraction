import cv2
import numpy as np
import common

color_infos=(0, 0, 255)
xmin=90
xmax=510
ymin=315
ymax=360
video='autoroute.mp4'

nbr_old=0
vehicule=0
seuil=10
# extraction de fon par soustraction en calculant la moyenne 
fond=common.moyenne_image(video, 500)
fond=fond[ymin:ymax, xmin:xmax]
cv2.imshow('fond', fond.astype(np.uint8))
fond=fond.astype(np.int32)
cap=cv2.VideoCapture(video)

while True:
    ret, frame=cap.read()
    # get unités de temps de l'horloge du CPU au debut de l'execution
    tickmark=cv2.getTickCount()
    # recuperation de mask  dans une region precis de frame RIO (Region of Interest)
    mask=common.calcul_mask(frame[ymin:ymax, xmin:xmax], fond, seuil)
    # détecter les contours des objets dans le masque binaire dans le cas ou une voiture passe
    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    nbr=0
    
    for e in elements:
        #  detection de le plus petit cercle englobant 
        # Cette fonction retourne deux valeurs : le centre du cercle (x, y) et le rayon du cercle
        ((x, y), rayon)=cv2.minEnclosingCircle(e)
        # si le rayon du cercle > 20 pixels => une voiture est passer 
        if rayon>20:
            # Le centre du cercle => (int(x) + xmin, int(y) + ymin) 
            cv2.circle(frame, (int(x)+xmin, int(y)+ymin), 5, color_infos, 10)
            nbr+=1
    if nbr>nbr_old:
        vehicule+=1
    nbr_old=nbr
    # Note1:
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}  Seuil: {:d}".format(fps, seuil), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_infos, 1)
    # dessiner un rectangle sur la region conserne 
    cv2.rectangle(frame, (xmin, ymin), (xmax+120, ymax), (255, 0, 0), 5)
    cv2.rectangle(frame, (xmax, ymin), (xmax+120, ymax), (255, 0, 0), cv2.FILLED)
    cv2.putText(frame, "{:04d}".format(vehicule), (xmax+10, ymin+35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)
    cv2.imshow('video', frame)
    cv2.imshow('mask', mask)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    # reajuster le seuil 
    if key==ord('p'):
        seuil+=1
    if key==ord('m'):
        seuil-=1

cap.release()
cv2.destroyAllWindows()


# Note 1:
"""
cv2.getTickFrequency() renvoie la fréquence de l'horloge. 
En divisant cela par la différence entre le nombre de "ticks" actuels (cv2.getTickCount()) 
et le nombre de "ticks" au début de l'analyse (tickmark), on obtient les FPS, c'est-à-dire 
le nombre de frames traitées par seconde.
"""
