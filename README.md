# Neural-Networks-Semester-Project
### Evan Ackerman
This is the repository for my semester project in CSE 40868.

# Part One: Conceptual Design
## Project Introduction and Practical Application
This semester, I will be attempting to build a neural network from scratch that will verify the authenticity of a handwritten signature. This project is actually quite applicable to the real world, most prominently with things like the mobile deposit of checks. The ease of depositing a check from your couch is amazing from the consumer’s perspective, but catching fraudulent checks still remains challenging for the banks. Another application is in the ever-so popular sports memorabilia industry. Validating and checking the signatures of the most famous athletes in the world is a profession that evaluates pieces of history worth millions of dollars, and an AI system can only help this industry even more. Forgery has always been a problem, and a neural network may be a nice solution to detecting this type of fraud.

## Working with and Acquiring Data
In terms of what is needed to be successful, the first task is acquiring the data. At glance, the provided data set from [Kaggle](https://www.kaggle.com/code/ananyapisal/starter-handwritten-signatures-c0f7b216-0/input) looks to have a decent amount of images for training/testing purposes. The first set in the link has 4 separate datasets with an equal number of actual and forged signatures. Based on this set alone, I think it would be best to train on 2 of the existing datasets from this particular source, and do validation on the 3rd, and testing on the 4th dataset. If this proves to not be enough data (or not be good for splitting appropriately), there are a bunch of other readily available sets of data. I can investigate other sources of data from Kaggle and potentially see if there is anything that is more conducive to the traditional 80/10/10 or 60/20/20 split for training, validation, and testing data.

One important note, because images are the mode of data, I have to ensure consistency in the image size/quality. The neural network should be trained with the same input size image, it may be very problematic (or catastrophic) for determining weights/biases if the images are not standardized. Also, with image sizes, depending on the route that is taken for this project, the images may need to be padded if a convolutional neural network is going to be used, as discussed in class briefly during Lecture 5! This will be something to definitely keep an eye out for as I begin to write code in the project and work with the data itself. 

To supplement my initial thoughts, I asked ChatGPT for some other potential things to look for or use in the data. It also brought up image size and to make sure there is a balance between  images large enough to capture detail, but also not too large to make the computation and weights manageable. Padding was also mentioned, as again I need to make sure that I am properly traversing the images and standardizing them. Other things ChatGPT mentioned that I didn’t think of is possibly doing data rotations for training to help increase the generalizability of the network. I will also eventually have to tune batch sizes, learning rates, etc, once I start training the network. This dialogue with ChatGPT was helpful to further expose myself to ways I can work with the data.

## Key Features of Signature Data
When thinking about the key features of the signature data, first and foremost the system will need to be able to detect what parts of the images are whitespace and what parts are part of the signature. Color should not matter for this data, as the specific pen color, etc, is not helpful in determining if the signature is real. The images should all just be grayscale, which will actually help training in the system since we will have less weights than if RGB values were used. While color is not important, the grayscale value still will be. The darkness of certain points in the signature can help identify where pens were picked up, put down, and spots that were overlapping. Curves and shapes of letters will be one of the most important features in the data to analyze, as letters with specific loops or patterns may be easily distinguishable for each signature. 

## First Blueprint/Thoughts for the Architecture of the System
Based on what I know from CSE 30124 and topics discussed so far in this course, my initial thought is going to be to design some sort of a convolutional neural network. My thought is to have some number of convolutional layers with non-linear activation functions, and also take advantage of pooling. My knowledge of pooling at this point, is that it will help the speed of the network by only taking the most important features, which will in turn lead to less total weights that need to be calculated. I also believe that the final output layer will only need one neuron, as the output of the system should just be one of two things: a valid or fraud signature. After reading the links given on Siamese networks, it seems like a CNN of this setup is a good way to go about the problem. By setting it up in parallel, the system can have a valid signature and potentially fraudulent signature and use the differences in the results to determine its validity. Admittedly, I need to return to those links to help fully understand Siamese architecture, but at this point in the semester I feel like I have a solid vision in place for moving forward.
