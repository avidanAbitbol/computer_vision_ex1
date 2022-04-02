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
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    return 302298963


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    if representation == 2:    
        #define & BGR -> RGB
        img_rgb = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        #normalize
        img_rgb = cv2.normalize(img_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        return img_rgb  
    else:
        #define & BGR -> RGB
        img_gray = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)  
        #define & RGB -> GRAY
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        #normalize
        img_gray = cv2.normalize(img_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        return img_gray   

    

def imDisplay(filename: str, representation: int):
    #using reader function to get the image and than display it
     img=imReadAndConvert(filename,representation)
     plt.imshow(img)
     plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    '''
    the math to get to yiq
    [Y] [0.299 0.587 0.114] [R]
    [I] = [0.596 -0.275 -0.321] [G]
    [Q] [0.212 -0.523 0.311] [B]
    '''
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]]).transpose()
     #doing the math equation to multiply it by the image
    img = np.dot(imgRGB,yiq_from_rgb)
    return img 


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
   rgb_from_yiq = np.array([[1,0.956, 0.619],
                           [1, -0.272, -0.647],
                           [1, -1.106, 1.703]]).transpose()
     #doing the math equation to multiply it by the image
   img = np.dot(imgYIQ,rgb_from_yiq)
   return img 

def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    
    if len(imgOrig.shape) == 3:  # checking if the image is RGB
        yiq_im = transformRGB2YIQ(imgOrig)  
        # do histogram Equalization on Y channel
        imgEq_y, histOrig, histEq = hsitogramEqualize_2D(yiq_im[:, :, 0]) 
        imgEq = np.array([imgEq_y, yiq_im[:, :, 1], yiq_im[:, :, 2]])
        imgEq = np.swapaxes(np.swapaxes(imgEq, 0, 1), 1, 2)
        imgEq = transformYIQ2RGB(imgEq)

    else:
        imgEq, histOrig, histEq = hsitogramEqualize_2D(imgOrig)

    return imgEq, histOrig, histEq


def hsitogramEqualize_2D(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    # set the image in the range [0,255]
    image_ = np.round(imgOrig * 255).astype(np.int) 
    #using hisogram function to build the plot
    histOrig = np.histogram(image_, bins=256)[0] 
    # create the look up table
    cumSum = np.cumsum(histOrig)
    CumSum_norm = cumSum / sum(histOrig)
    LUT = np.round(CumSum_norm * 255)
    # do the Equalization on the image and normalize back to [0,1]
    imgEq = (np.array(LUT[image_.flatten()]).reshape(imgOrig.shape)) / 255
    histEq = np.histogram(np.round(imgEq*255), bins=256)[0]

    return imgEq, histOrig, histEq



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    if len(imOrig.shape) == 3:
         img = transformRGB2YIQ(imOrig) * 255
         #taking the y channel
         histOrig , bins = np.histogram(img[:, :, 0].flatten(), 256, [0, 256])
         z_arr = np.arange(nQuant + 1)  # with 0 and 255
         q_arr = np.arange(nQuant)
         for i in z_arr:  # init
             z_arr[i] = round((i / nQuant) * len(histOrig - 1))

         img_list = []
         mse_list = []
         for k in range(0, nIter):
             for i in range(0, nQuant):
#Finding q - the values that each of the segments’ intensities will map to. 
#q is also a vector, however,containing nQuant elements.
                 q_arr[i] = np.average(bins[z_arr[i]:z_arr[i + 1]].reshape(1, -1),
                                   weights=histOrig[z_arr[i]:z_arr[i + 1]].reshape(1, -1))
             for j in range(1, nQuant):
#Finding z - the borders which divide the histograms into segments.
# z is a vector containing nQuant+1elements. The first and last elements are 0 and 255 respectively.
                 z_arr[j] = round((q_arr[j - 1] + q_arr[j]) / 2)
             newimg = img.copy()
             for i in range(1, nQuant + 1):
                 newimg[:, :, 0][(newimg[:, :, 0] > z_arr[i - 1]) & (newimg[:, :, 0] < z_arr[i])] = q_arr[i - 1]
             newimg = transformYIQ2RGB(newimg) / 255
#making a list of the quantized image in each iteration
             img_list.append(newimg)
#calculate the mse(mean squared error)
             mse=pow(np.power(imOrig-newimg,2).sum(),0.5)/imOrig.size
#making list of the MSE error in each iteration
             mse_list.append(mse)
#
    else:
        img = imOrig * 255
        histOrig, bins = np.histogram(img.flatten(), 256)
        bins = np.arange(0, 256)
        z_arr = np.arange(nQuant + 1)  # with 0 and 255
        q_arr = np.arange(nQuant, dtype=np.float32)
        for i in range(0, len(z_arr)):  #start
            z_arr[i] = round((i / nQuant) * len(histOrig))
            img_list = []
            mse_list = []
        for k in range(0, nIter):
            for i in range(0, nQuant):
#Finding q - the values that each of the segments’ intensities will map to. 
#q is also a vector, however,containing nQuant elements.
                q_arr[i] = np.average(bins[z_arr[i]:z_arr[i + 1] + 1], weights=histOrig[z_arr[i]:z_arr[i + 1] + 1])
            for j in range(1, nQuant):
#Finding z - the borders which divide the histograms into segments.
# z is a vector containing nQuant+1elements. The first and last elements are 0 and 255 respectively.
                z_arr[j] = (q_arr[j - 1] + q_arr[j]) / 2
            newimg = img.copy()
            for i in range(1, nQuant + 1):
                #making a list of the quantized image in each iteration
                newimg[(newimg >= z_arr[i - 1]) & (newimg < z_arr[i])] = q_arr[i - 1]
            img_list.append(newimg)
            #calculate the mse
            mse = pow(np.power(img - newimg, 2).sum(), 0.5) / img.size
            #making list of the MSE error in each iteration
            mse_list.append(mse)

    return img_list, mse_list

    pass
