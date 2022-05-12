# Image-Transformation
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:Import the necessary libraries and read the original image and save it a image variable.
<br>

### Step2:Translate the image using Translation_matrix=np.float32([[1,0,120],[0,1,120],[0,0,1]]) Translated_image=cv2.warpPerspective(org_img,Translation_matrix,(col,row))
<br>

### Step3:Scale the image using Scaling_Matrix=np.float32([[1.2,0,0],[0,1.2,0],[0,0,1]]) Scaled_image=cv2.warpPerspective(org_img,Scaling_Matrix,(col,row))
<br>

### Step4:Shear the image using

Shearing_matrix=np.float32([[1,0.2,0],[0.2,1,0],[0,0,1]]) Sheared_image=cv2.warpPerspective(org_img,Shearing_matrix,(col2,int(row1.5)))
<br>

### Step5:Reflection of image can be achieved through the code Reflection_matrix_row=np.float32([[1,0,0],[0,-1,row],[0,0,1]]) Reflected_image_row=cv2.warpPerspective(org_img,Reflection_matrix_row,(col,int(row)))
<br>
### Step6:Rotate the image using Rotation_angle=np.radians(10) Rotation_matrix=np.float32([[np.cos(Rotation_angle),-np.sin(Rotation_angle),0], [np.sin(Rotation_angle),np.cos(Rotation_angle),0], [0,0,1]]) Rotated_image=cv2.warpPerspective(org_img,Rotation_matrix,(col,(row)))


<br>
### Step7:Crop the image using cropped_image=org_img[10:350,320:560]
<br>
### Step8:Display all the Transformed images.
<br>

## Program:
```python
Developed By: A.sasidharan
Register Number: 212221240049
i)Image Translation
import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("dog.jpg")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
M= np.float32([[1, 0, 100],
                [0, 1, 200],
                 [0, 0, 1]])
translatedImage =cv2.warpPerspective (inputImage, M, (cols, rows))
plt.imshow(translatedImage)
plt.show()


ii) Image Scaling
import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("dog.jpg")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
M = np. float32 ([[1.5, 0 ,0],
                 [0, 1.8, 0],
                  [0, 0, 1]])
scaledImage=cv2.warpPerspective(inputImage, M, (cols * 2, rows * 2))
plt.imshow(scaledImage)
plt.show()


iii)Image shearing
import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("dog.jpg")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
matrixX = np.float32([[1, 0.5, 0],
                      [0, 1 ,0],
                      [0, 0, 1]])

matrixY = np.float32([[1, 0, 0],
                      [0.5, 1, 0],
                      [0, 0, 1]])
shearedXaxis = cv2.warpPerspective (inputImage, matrixX, (int(cols * 1.5), int (rows * 1.5)))
shearedYaxis = cv2.warpPerspective (inputImage, matrixY, (int (cols * 1.5), int (rows * 1.5)))
plt.imshow(shearedXaxis)
plt.show()
plt.imshow(shearedYaxis)
plt.show()


iv)Image Reflection

import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("dog.jpg")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
matrixx=np.float32([[1, 0, 0],
                    [0,-1,rows],
                    [0,0,1]])
matrixy=np.float32([[-1, 0, cols],
                    [0,1,0],
                    [0,0,1]])
reflectedX=cv2.warpPerspective(inputImage, matrixx, (cols, rows))
reflectedY=cv2.warpPerspective(inputImage, matrixy, (cols, rows))
plt.imshow(reflectedY)
plt.show()


v)Image Rotation

import cv2
import numpy as np
import matplotlib.pyplot as plt
angle=np.radians(45)
inputImage=cv2.imread("dog.jpg")
rows, cols, dim = inputImage.shape
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],
               [np.sin(angle),np.cos(angle),0],
               [0,0,1]])
rotatedImage = cv2.warpPerspective(inputImage,M,(int(cols),int(rows)))
plt.axis('off')
plt.imshow(rotatedImage)
plt.show()


vi)Image Cropping
import cv2
import numpy as np
import matplotlib.pyplot as plt
angle=np.radians(45)
inputImage=cv2.imread("dog.jpg")
CroppedImage= inputImage[20:150, 60:230]
plt.axis('off')
plt.imshow(CroppedImage)
plt.show()

```
## Output:
### i)Image Translation
<br>![sasi 1](https://user-images.githubusercontent.com/94154712/168055090-9495e3a5-2992-48b6-a69d-883d80025216.png)

<br>
<br>
<br>

### ii) Image Scaling
<br>![sasi 2](https://user-images.githubusercontent.com/94154712/168055054-18814a16-7989-4827-a20d-6947b8dc7728.png)

<br>
<br>
<br>


### iii)Image shearing
<br>![sasi 3](https://user-images.githubusercontent.com/94154712/168055008-b4933d29-6493-436b-a92c-374204e4494e.png)

<br>
<br>
<br>


### iv)Image Reflection
<br>![sasi 4](https://user-images.githubusercontent.com/94154712/168054949-d9f9004f-7cd6-4b8c-93eb-19ecb9706d89.png)

<br>
<br>
<br>



### v)Image Rotation
<br>![sasi 5](https://user-images.githubusercontent.com/94154712/168053581-816df14b-e1a6-4861-9955-8414ad793226.png)

<br>
<br>
<br>



### vi)Image Cropping
<br>![sasi 6](https://user-images.githubusercontent.com/94154712/168053692-d4151962-9bdb-474e-a900-44e021cfd4ad.png)

<br>
<br>
<br>




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
