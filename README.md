# Gender-and-Age-Detection   <img alt="GitHub" src="https://img.shields.io/github/license/smahesh29/Gender-and-Age-Detection">

<h2>Objective :</h2>
<p>To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.</p>

<h2>About the Project :</h2>
<p>In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html">Tal Hassner and Gil Levi</a>. The predicted gender may be one of 'Male' and 'Female', and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.</p>

<h2>Dataset :</h2>
<p>For this python project, I had used the Adience dataset; the dataset is available in the public domain and you can find it <a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification">here</a>. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models I used had been trained on this dataset.</p>

<h2>Additional Python Libraries Required :</h2>
<ul>
  <li>OpenCV</li>
  
       pip install opencv-python
</ul>
<ul>
 <li>argparse</li>
  
       pip install argparse
</ul>

<h2>The contents of this Project :</h2>
<ul>
  <li>opencv_face_detector.pbtxt</li>
  <li>opencv_face_detector_uint8.pb</li>
  <li>age_deploy.prototxt</li>
  <li>age_net.caffemodel</li>
  <li>gender_deploy.prototxt</li>
  <li>gender_net.caffemodel</li>
  <li>a few pictures to try the project on</li>
  <li>detect.py</li>
 </ul>
 
<h2>Usage :</h2>
<ul>
  <li>Download my Repository</li>
  <li>Open your Command Prompt or Terminal and change directory to the folder where all the files are present.</li>
  <li><b>Detecting Gender and Age of face in Image</b> Use Command :</li>
  
      python detect.py --image <image_name>
</ul>
  
<h2>Examples :</h2>

<p><b>NOTE:- I downloaded the images from Google. You can use any image for testing.</b></p>

<img src="Example/Detecting age and gender girl1.png">
<img src="Example/Detecting age and gender girl2.png">
<img src="Example/Detecting age and gender kid1.png">
<img src="Example/Detecting age and gender kid2.png">
<img src="Example/Detecting age and gender man1.png">
<img src="Example/Detecting age and gender man2.png">
<img src="Example/Detecting age and gender woman1.png">

<h2>Improvements in this Version :</h2>
<ul>
  <li>Added rate limiting and retry mechanism to handle resource exhaustion</li>
  <li>Improved error handling and user feedback</li>
  <li>Added smooth video processing with frame skipping for better performance</li>
  <li>Enhanced age estimation with middle values for more precise predictions</li>
  <li>Better webcam mode with continuous display of last valid detection</li>
</ul>

<h2>Working :</h2>
<p>This Python Project uses OpenCV Deep Learning models to detect age and gender. The models used in this project were trained by Tal Hassner and Gil Levi. The predicted gender may be one of 'Male' and 'Female', and the predicted age may be one of the ranges (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100).</p>

<h2>Support :</h2>
<p>If you encounter any issues or have questions, please open an issue in the repository.</p>
