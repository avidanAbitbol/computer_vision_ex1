# computer_vision_ex1
This is an exrecise to build first skills in computer vision field ,the function's we were require to implement are:
## 1. Reading an image into a given representation –

I use functions from cv2 to use the numpy array and represent the images by RGB\grey scale,I used normalize function to normalize the array between 0-1.
## 2. Displaying an image:

I used the convert function I built and used imshow & show function to show the image.
## 3. Transforming an RGB image to YIQ color space –

I build an array that present yiq and multiply it by the original array to get a new image.
## 4. Histogram equalization:

performs histogram equalization of a given grayscale or RGB image
## 5. Optimal image quantization:

performs optimal quantization of a given grayscale or RGB image
## 6. Gamma Correction:

I show the new image by multiply the number with the image to get the new correction ,this question was made with gui to play with the number you multiply the image to get the new correction.

# Test

i run some personal photos to test the function's i build:
## from original to yiq(left original)

![image](https://user-images.githubusercontent.com/80645472/161398326-915f553f-a308-47a8-a28c-48db329bf87d.png)
## from yiq to original(left original)

![image](https://user-images.githubusercontent.com/80645472/161398331-a8ac8170-a7db-4bd7-859f-99da85ea5c52.png)
## Histogram equalization in gray(right side) and RGB(left side)

![image](https://user-images.githubusercontent.com/80645472/161398367-e082b14b-22c4-40ec-893d-ac3f0033c751.png)
##Optimal image quantization

![image](https://user-images.githubusercontent.com/80645472/161399404-53b52dc7-25e0-459c-8234-3ecf5425280f.png)

![image](https://user-images.githubusercontent.com/80645472/161399462-74915d4d-f177-4223-894b-60e254435652.png)


## Gamma Correction

![image](https://user-images.githubusercontent.com/80645472/161398406-6cef299c-0b47-4f97-990d-0f3a02723f51.png)

## list of files in this project:

![image](https://user-images.githubusercontent.com/80645472/161399767-c0b229b5-3dc8-4a38-aa17-02f117e4d31b.png)

# *This project was done on "spyder" "python 3.9" version.*
