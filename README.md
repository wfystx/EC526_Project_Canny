### EC526 Final Project Edge Detection with Canny algorithm
- Canny edge detection is a technique to **extract useful structural information** from different vision objects and dramatically reduce the amount of data to be processed. It has been widely applied in various computer vision systems.

### The general criteria for edge detection include: 
- Detection of edge with low error rate, which means that the detection should accurately catch as many edges shown in the image as possible. 
- The edge point detected from the operator should accurately localize on the center of the edge.
- A given edge in the image should only be marked once, and where possible, image noise should not create false edges. To satisfy these requirements Canny used the calculus of variations – a technique which finds the function which optimizes a given functional.

### Environment:
- OpenCV

- C++ 11

- OpenMP

- Visual Studio 2017

### Usage
- [Install the Visual Studio Correctly.](https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019)
- Set OpenMP and OpenCV environment and link the to the Visual Studio 2017.["opencv"](https://docs.opencv.org/master/dd/d6e/tutorial_windows_visual_studio_opencv.html) ["openMP"](https://docs.microsoft.com/en-us/cpp/parallel/openmp/openmp-in-visual-cpp?view=vs-2019)
- Build the executive file for windows and processing the image
![image](https://github.com/wfystx/EC526_Project_Canny/blob/master/Documents/Readme_Images/output.PNG) 
### Original Picture

![image](https://github.com/wfystx/EC526_Project_Canny/blob/master/Documents/Readme_Images/phanSneeze.jpg) 

### Outputs
![image](https://github.com/wfystx/EC526_Project_Canny/blob/master/Documents/Readme_Images/general_1.PNG) 
![image](https://github.com/wfystx/EC526_Project_Canny/blob/master/Documents/Readme_Images/general_2.PNG) 
![image](https://github.com/wfystx/EC526_Project_Canny/blob/master/Documents/Readme_Images/general_3.PNG) 
![image](https://github.com/wfystx/EC526_Project_Canny/blob/master/Documents/Readme_Images/sobel_prewitt.PNG) 
![image](https://github.com/wfystx/EC526_Project_Canny/blob/master/Documents/Readme_Images/Sobel_Robert.PNG) 
![image](https://github.com/wfystx/EC526_Project_Canny/blob/master/Documents/Readme_Images/table.PNG) 

### Authors

* **Mingdao Che** 
* **Fuyao Wang** 
* **Pat Rick Ntwari**
