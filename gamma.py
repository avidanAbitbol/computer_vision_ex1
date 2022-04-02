"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np

def gamma(x):
    return

#
def gammaDisplay(img_path: str, rep: int):
    img = cv2.imread(img_path, rep - 1)
    cv2.namedWindow('Gamma correction')
    #creating the track bar for gui and manipulat between 0-2 by doing it 0-200
    cv2.createTrackbar('Gamma', 'Gamma correction', 1, 200, gamma)
    img = np.asarray(img)/255
    #using the image and show it
    cv2.imshow('Gamma correction', img)
    term = cv2.waitKey(1)
    newim = img
    #to get it run over the playing time 
    while 1:
        cv2.imshow('Gamma correction', newim)
        term = cv2.waitKey(1) & 0xFF
        if term == 27:
            break
        gamma_cor = cv2.getTrackbarPos('Gamma', 'Gamma correction')
        #the math manipulate
        newim = np.power(img, gamma_cor/100)
    cv2.destroyAllWindows()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)
    
    #gammaDisplay('images2.jpg', 2)


if __name__ == '__main__':
    main()
