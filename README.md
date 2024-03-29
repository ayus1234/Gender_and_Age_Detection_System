# Gender-and-Age-Detection   <img alt="GitHub" src="https://img.shields.io/github/license/smahesh29/Gender-and-Age-Detection">


<h2>Objective :</h2>
<p>To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.</p>

<h2>About the Project :</h2>
<p>In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html">Tal Hassner and Gil Levi</a>. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.</p>

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
 <p>For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.</p>
 
 <h2>Usage :</h2>
 <ul>
  <li>Download my Repository</li>
  <li>Open your Command Prompt or Terminal and change directory to the folder where all the files are present.</li>
  <li><b>Detecting Gender and Age of face in Image</b> Use Command :</li>
  
      python detect.py --image <image_name>
</ul>
  <p><b>Note: </b>The Image should be present in same folder where all the files are present</p> 
<ul>
  <li><b>Detecting Gender and Age of face through webcam</b> Use Command :</li>
  
      python detect.py
</ul>
<ul>
  <li>Press <b>Ctrl + C</b> to stop the program execution.</li>
</ul>

<h2>Examples :</h2>
<p><b>NOTE:- I downloaded the images from Google,if you have any query or problem i can remove them, i just used it for Educational purpose.</b></p>

    >python detect.py --image girl1.jpg
    Gender: Female
    Age: 25-32 years
    
![girl1](https://github.com/ayus1234/Gender_and_Age_Detection_System/assets/107507481/a8b72998-0951-4f19-b923-68cc000a0002)


    >python detect.py --image girl2.jpg
    Gender: Female
    Age: 8-12 years
    
![girl2](https://github.com/ayus1234/Gender_and_Age_Detection_System/assets/107507481/7a41e344-5593-4bb5-8924-f427220e1054)

    >python detect.py --image kid1.jpg
    Gender: Male
    Age: 4-6 years    
    
![kid1](https://github.com/ayus1234/Gender_and_Age_Detection_System/assets/107507481/a9408b05-385e-4ee0-8448-652ea766a0aa)


    >python detect.py --image kid2.jpg
    Gender: Female
    Age: 4-6 years  
    
![kid2](https://github.com/ayus1234/Gender_and_Age_Detection_System/assets/107507481/6e93b943-6f61-4047-9137-8c61abc3a0c8)

    >python detect.py --image man1.jpg
    Gender: Male
    Age: 38-43 years
    
![man1](https://github.com/ayus1234/Gender_and_Age_Detection_System/assets/107507481/d1746989-2bdb-4c5e-8a2d-f0a1a01225f3)

    >python detect.py --image man2.jpg
    Gender: Male
    Age: 25-32 years
    
![man2](https://github.com/ayus1234/Gender_and_Age_Detection_System/assets/107507481/ebd37f4e-d5c4-40b2-955e-5ede55cc4ad0)

    >python detect.py --image woman1.jpg
    Gender: Female
    Age: 38-43 years
    
![woman1](https://github.com/ayus1234/Gender_and_Age_Detection_System/assets/107507481/9089c134-00df-486b-9023-e064719e7f2c)
         
