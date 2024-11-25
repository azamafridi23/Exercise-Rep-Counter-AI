
# Exercise Rep Counter AI 🤖

This project leverages advanced AI 🤖 and computer vision 👁️ to detect exercises in real-time, counting repetitions based on webcam footage 📷. By utilizing OpenCV for camera access, TensorFlow/Keras for custom deep learning models, and MediaPipe for human pose estimation, the system accurately recognizes exercises, tracks repetitions, and provides insightful visualizations such as joint angles and exercise probabilities. Designed for ease of use, this project can be seamlessly run on a local machine, with clear installation and setup instructions.

## Demo
- Video:



https://github.com/user-attachments/assets/31fe381c-61c9-41b7-9dd1-e60f9a5c99e4



### Features
This project leverages:

- Real-time Exercise Detection: Uses webcam footage to detect and count exercise repetitions.
- Pose Estimation: Powered by Google MediaPipe's BlazePose model for accurate human pose estimation.
- Custom Deep Learning Models: Built with TensorFlow/Keras for exercise classification.
- Accurate Rep Counting: Tracks exercise movements and counts reps based on joint angles.
- Interactive Visualizations: Displays joint angles, rep counters, and probability distributions.
- Guided Data Collection: A pipeline for generating custom training data tailored for your use case.

### AI Model Performance

- LSTM Model:
    - Accuracy: 97.78%
    - Categorical Cross-Entropy Loss: 1.51e-3 (Validation dataset)
- Attention-Based LSTM:
    - Accuracy: 99.1%
    - Categorical Cross-Entropy Loss: 2.08e-5 (Validation dataset)

#### Technologies Used

- OpenCV: For webcam access, camera properties, color conversion, and image display.
- TensorFlow/Keras: For building and training the custom deep learning models.
- Streamlit: For creating an interactive web app to use the AI model without needing web development expertise.
- Google MediaPipe: For real-time pose estimation and joint angle calculation.
- Jupyter Notebook: For data analysis and model training.

#### Pose Tracking Full Body Landmarks:
![pose_tracking_full_body_landmarks.png](pose_tracking_full_body_landmarks.png)


### Installation

- Download this repository and move it to your desired working directory
- Download Anaconda if you haven't already
- Open the Anaconda Prompt
- Navigate to your working directory using the cd command
- Run the following command in the Anaconda prompt:
	```
  	conda env create --name NAME --file environment.yml
  	```
	where NAME needs to be changed to the name of the conda virtual environment for this project. This environment contains all the package installations and dependencies for this project.
  
- Run the following command in the Anaconda prompt:
  	```
  	conda activate NAME
  	```
	This activates the conda environment containing all the required packages and their versions. 
  
- Open Anaconda Navigator
- Under the "Applications On" dropdown menu, select the newly created conda environment
- Install and open Jupyter Notebook. NOTE: once you complete this step and if you're on a Windows device, you can call the installed version of Jupyter Notebook within the conda environment directly from the start menu.  
- Navigate to the Training Notebook.ipynb file within the repository

### Features in Detail

1. Human Pose Estimation (BlazePose)
The project integrates BlazePose, a model from Google MediaPipe, to detect human pose and extract key body landmarks from the webcam footage.

2. Deep Learning Models
Custom LSTM Model: Used to classify exercises based on the detected body pose.
Attention-Based LSTM: Improves accuracy by focusing on specific pose features during exercise classification.
3. Real-time Rep Counting
Using joint angles computed from the body landmarks, the system tracks and counts the repetitions during an exercise.

4. Data Collection and Training Pipeline
A guided process for generating labeled training data, which is essential for training the custom models. The pipeline simplifies data collection and preprocessing for deep learning.

5. Visualization Tools
	- Visualize the exercise performance in real-time with:
	- Joint angle graphs.Rep counters.
	- Probability distributions showing the likelihood of the detected exercise.
