# Senior Design: Attendance MLP with SVD
Welcome to the senior design project that utilizes Multilayered Perceptron (MLP) and Singular Value Decomposition (SVD). The purpose of this project is for automatted attendance and utilizing neural networks without premade neural network libraries. This project currently only works for one class. If you would like a summary of each folder and file, skip to the File and Folder Overview section.
## Setup 
1. Download [Anaconda](https://www.anaconda.com/download/success) and step through the basic installation process.
2. Download [Python](https://www.python.org/downloads/) and step through the basic installation process.
3. Run set_conda.bat by double clicking on the script. Follow the scripts instructions.
4. Open command prompt and enter the following lines in order
   - ```conda create -n "base"```
   - ```conda activate base```
   - Find the file location of SKYNET-VISION and type in ```cd [SKYNET-VISION folder location]/src```
   - ```python setup.py```
   - ```conda deactivate```
## Running the Program
There are two options you have when running this program; run through command prompts or running *.bat scripts. Please note, that *.bat scripts require a Windows machine. Running the command prompt lines will work in both VS Code and the basic command prompt for all computers. The File and Folder Overview section gives the basics for what each script does. The following steps are the proper way to run the program:
1. Collect 10 to 15 photos from each student in a classroom
2. Create folders with each student's name in the Students/Uncropped_Students folder and store the photo of each student in their corresponding folders.
3. Train a model: Run the train_model.bat file or run the following code in a command prompt/VS Code:
    - ```conda activate base``` (Not necessary if using VS Code and you have setup a python environment)
    - Find the file location of SKYNET-VISION and type in ```cd [SKYNET-VISION folder location]/src```
    - ```python train.py```
4. Put photos that you would like to take attendance of in the Attendance_Photos folder
5. Take attendance: Run the attendance.bat file or run the follwoing code in a command prompt/VS Code:
    - ```conda activate base``` (Not necessary if using VS Code and you have setup a python environment)
    - Find the file location of SKYNET-VISION and type in ```cd [SKYNET-VISION folder location]/src```
    - ```python main.py```
6. Repeat Steps 4 and 5 if you want to take attendance again.

## File and Folder Overview
This section will cover the basics of each file and folder on this level of folder organization.
##### Attendance_Photos
The Attendance_Photos folder is where you will put photos of people that you like to identify. After training the model, you put photos of your class in this folder and the program will identify all the faces in all the photos, and then take attendance of the class. All of the faces that are identified in these photos will be cropped, grayscaled, and stored in the Identifed_Faces folder automatically.
##### Identified_Faces
The Identified_Faces folder stores the processed faces found from the photos in Attendance_Photos. The only time that a user needs to go into this folder is after the program has found all the faces and is asked to make sure there are no misidentifed photos. It is a common for the program to identify a portion of a wall as a face, and those photos need to be removed before attendance is taken.
##### src
The src folder stores all of the program files for this project. It also includes a storage of neural network models, and SVD dataset files. A user shouldn't have this folder unless they would like to run the program manually or just look through the code. There is an additional README.md that summarizes all the files in this folder.
##### Students
The Students folder is where the user sets up the database for their class. There are two subfolders underneath the Students folder, the Cropped_Students and Uncropped_Students folders. To make a database for your class you need to make folders titled with your students in the Uncropped_Students folder. Each of your student's folders is where you will store 10 to 15 photos for those corresponding students. The program will go through all of the students in Uncropped_Students and store/crop all the photos into the Cropped_Students folder. The user will be asked to go through the Cropped_Students folder and delete all bad photos from those folders before training is done.
##### *.bat files
The *.bat files are attempts at automating the process for using the code. The set_conda.bat will make it so the Anaconda, a python environment, can be utilizied in a command prompts. Attendance.bat runs the attendance code, while train_model.bat run the training model code. This *.bat files are only necessary if you don't use the terminal commands listed above.  
