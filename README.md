# Neural-Networks-Semester-Project
### Evan Ackerman and Patrick Schlosser
This is the repository for our semester project in CSE 40868.
Spring 2024

# Part One: Conceptual Design
## Project Introduction and Practical Application
This semester, we will be attempting to build a neural network from scratch that will verify the authenticity of a handwritten signature. This project is actually quite applicable to the real world, most prominently with things like the mobile deposit of checks. The ease of depositing a check from your couch is amazing from the consumer’s perspective, but catching fraudulent checks still remains challenging for the banks. Another application is in the ever-so popular sports memorabilia industry. Validating and checking the signatures of the most famous athletes in the world is a profession that evaluates pieces of history worth millions of dollars, and an AI system can only help this industry even more. Forgery has always been a problem, and a neural network may be a nice solution to detecting this type of fraud.

## Evan's Initial Thoughts

## Working with and Acquiring Data
In terms of what is needed to be successful, the first task is acquiring the data. At glance, the provided data set from [Kaggle](https://www.kaggle.com/code/ananyapisal/starter-handwritten-signatures-c0f7b216-0/input) looks to have a decent amount of images for training/testing purposes. The first set in the link has 4 separate datasets with an equal number of actual and forged signatures. Based on this set alone, I think it would be best to train on 2 of the existing datasets from this particular source, and do validation on the 3rd, and testing on the 4th dataset. If this proves to not be enough data (or not be good for splitting appropriately), there are a bunch of other readily available sets of data. I can investigate other sources of data from Kaggle and potentially see if there is anything that is more conducive to the traditional 80/10/10 or 60/20/20 split for training, validation, and testing data.

One important note, because images are the mode of data, we have to ensure consistency in the image size/quality. The neural network should be trained with the same input size image, it may be very problematic (or catastrophic) for determining weights/biases if the images are not standardized. Also, with image sizes, depending on the route that is taken for this project, the images may need to be padded if a convolutional neural network is going to be used, as discussed in class briefly during Lecture 5! This will be something to definitely keep an eye out for as we begin to write code in the project and work with the data itself. 

To supplement my initial thoughts, I asked ChatGPT for some other potential things to look for or use in the data. It also brought up image size and to make sure there is a balance between images large enough to capture detail, but also not too large to make the computation and weights manageable. Padding was also mentioned, as again I need to make sure that I am properly traversing the images and standardizing them. Other things ChatGPT mentioned that I didn’t think of is possibly doing data rotations for training to help increase the generalizability of the network. I will also eventually have to tune batch sizes, learning rates, etc, once I start training the network. This dialogue with ChatGPT was helpful to further expose myself to ways I can work with the data.

## Key Features of Signature Data
When thinking about the key features of the signature data, first and foremost the system will need to be able to detect what parts of the images are whitespace and what parts are part of the signature. Color should not matter for this data, as the specific pen color, etc, is not helpful in determining if the signature is real. The images should all just be grayscale, which will actually help training in the system since we will have less weights than if RGB values were used. While color is not important, the grayscale value still will be. The darkness of certain points in the signature can help identify where pens were picked up, put down, and spots that were overlapping. Curves and shapes of letters will be one of the most important features in the data to analyze, as letters with specific loops or patterns may be easily distinguishable for each signature. 

## First Blueprint/Thoughts for the Architecture of the System
Based on what I know from CSE 30124 and topics discussed so far in this course, my initial thought is going to be to design some sort of a convolutional neural network. My thought is to have some number of convolutional layers with non-linear activation functions, and also take advantage of pooling. My knowledge of pooling at this point, is that it will help the speed of the network by only taking the most important features, which will in turn lead to less total weights that need to be calculated. I also believe that the final output layer will only need one neuron, as the output of the system should just be one of two things: a valid or fraud signature. After reading the links given on Siamese networks, it seems like a CNN of this setup is a good way to go about the problem. By setting it up in parallel, the system can have a valid signature and potentially fraudulent signature and use the differences in the results to determine its validity. Admittedly, I need to return to those links to help fully understand Siamese architecture, but at this point in the semester I feel like we have a solid vision in place for moving forward.


## Patrick's Thoughts

## Data Acquisition
Additional datasets we could consider working with are the [CEDAR Signature Database](https://paperswithcode.com/dataset/cedar-signature) and the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). Using data from multiple different sources may also help us eliminate any bias in the presentation format of that data. Seeing data in slightly different forms could also help the model's performance when presented with images written on a different surface or with a different writing utensil. This could also help us reach a training, validation, and testing split closer to 70%/15%/15%. 

## Prominent Data Features
Differentiated datasets may also help us avoid performing classifications based on accidents. Signatures and handwriting data will have many essential features such as character spacing, intensity, proportions, size, and shape (curves, loops, and crosses) as well as the characters themselves, the overall writing speed, and the beginning and ending strokes to name a few; there are many accidental features we should avoid using as a basis of classification, however, such as text color or orientation and image size. Our model should not account for where the text appears in frame or how the paper appears under the text. In order to account for these possible differences we should consider performing data preprocessing and standardization such as resizing images, converting images to grayscale, or even converting them to strictly black and white (background removal). 

## Model Architecture Selection
Essentially, the task of signature verification is to identify differences between similar images in order to group signatures together and maximize the distance from forgeries. We can consider a few different architectures for this task such as Siamese Networks, Triplet Networks, or Contrastive Loss Networks. A Contrastive Loss Network may be more computationally efficient since no duplication of architecture is required, but it also may be less flexible in learning complex feature representation. A Siamese Network's twin subnetworks combined with Euclidian distance or Cosine differentiator may allow us to analyze more subtle differences. If our image preprocessing is less successful than we would hope, we could also consider implementing some sort of attention mechanism in our design. These features could be built using TensorFlow or PyTorch.

## Training and Evaluation
The training process will most likely utilize mini-batch gradient descent and backpropagation. We will need to evaluate the computational power we have available and our desired training dataset size to estimate the training duration. To optimize the training process we will need to determine the best hyperparameter settings including learning rate and backpropagation, and we can monitor the process' performance with the validation dataset to avoid overfitting to peculiarities of the training set. We can then evaluate the performance of our model using our testing dataset and typical evaluation metrics. For handwriting recognition, we should in particular examine the false acceptance rate and the false rejection rate. For practical purposes, we should in particular try to minimize false rejections. 

## Optimization and Deployment
After the initial training process is complete, if we have time we may find a need to adjust some aspects of our model such as preprocessing, model architecture, or the hyperparameters. If we are struggling with our datasets, we could also play around with data augmentation techniques. Once our network is completed we will hopefully be able to use it to correctly verify our own signatures and deny forged attempts. 

_ChatGPT was used for reference in the above._

# Part Two: Datasets
With a vision in mind from Part One, we are now onto working with data so that we can create this network that will be able to detect fraud for any signature.

## Source of Data
#### Evan
The source of the data is the [Kaggle](https://www.kaggle.com/code/ananyapisal/starter-handwritten-signatures-c0f7b216-0/input) set provided in the project description page. There are 30 subjects in this data set, with 5 real signatures and 5 forged signatures each. In essence, we have about 300 raw points. However, in order to maximize utility out of this dataset, we will also perform mirroring on each of the signatures to allow us to have more things to input. We will also each add a sample of our own signatures to the test set after our model has been trained. That would bring us to 32 subjects, with 10 signatures each, and each having 4 versions of that signature with the different mirrors.

## Differences in Trainings/Validation Subsets
#### Patrick
After aquiring our initial dataset, a challenge we initially faced was to differentiate between unique images and duplicate images. The dataset iteslf was divided into four subsections, but most of the data between each section was duplicated. After parsing the data to identify unique samples, we chose to initially focus our training and validation sets on the Kaggle data. We chose to use samples from completely different signees for training and validation data to identify possible areas of our model overfitting to individual writing styles. Our training data will contain 110 unique raw images (paired with 110 forged so 220 total) and our validation data will contain 25 unique raw images (paired with 25 forged so 50 total) so our validation subset will be about 22.7% of the size of our training data. The total breakdown based on just the Kaggle set would then be 73/16/11, however at the end we will be adding our own signatures to the testing set, so the breakdown will get more spaced out.

## Number of Distinct Subjects/Samples per Subject
#### Patrick & Evan
As mentioned earlier, there are a total of 30 subjects in this dataset with 5 real and 5 forged signatures for each. To increase the size of our training, validation, and testing data, we will include not only the original images, but also a vertical reflection, a horizontal reflection, and a vertically and horizontally reflected copy for each sample which will increase the size of each data subset by a factor of 4. This means that we actually use 440 inputs (paired with equal number of forged) for training and 100 inputs (paired with an equal number of forged) for validation. In addition to the 15 unique paired photos from the kaggle data we set aside for testing, we will include 5 of each of our own true and forged signatures in the testing set which will give 25 unique photos (paired with forged) and 100 inputs with rotation for the testing set. We made a diagram to help visualize the breakdown of our data with subjects/samples:

![diagram](https://github.com/eackerm2/Neural-Networks-Semester-Project/assets/122949257/ebbad0da-180a-4b26-b287-0d6bc726b374)


## Characterization of Samples
#### Patrick
Among all of the images in the data set they can be broken down by: \
Approximate Image Width Range: 150 - 1700 pixels \
Approximate Image Height Range: 50 - 750 pixels \
Approximate Image Bit Depth: 8 or 24 \
File Type: PNG \
Clearly, our input data is not standardized in any way so we will need to do so. The images come in a wide range of file sizes, heights, widths, and colors. Some images are already grayscale and some need to be converted to grayscale. We will also scale all images to 200x75 pixels for standardized input. All images from the Kaggle dataset are in PNG format. 

## Example Samples
#### Evan
After getting into the dataset and working in CoLab, we were able to start some initial pre-processing. The first step of pre-processing was to convert all of the images to grayscale. We don't care about the color of the signature, the most important parts are the changes in saturation (which indicate where pens may have been picked up etc) and the straightness of the strokes. By converting all to grayscale, we are making sure everything is standardized going forward. Another pre-processing item we performed was the addition of mirroring, this way for each signature there can be more points for us to train with. An example signature after converted to grayscale with its respective rotations can be seen below (Images not to scale in this photo, merely screenshots from CoLab). Tentatively we have resized to 200x75 in our code, as we feel like this won't be too small to lose detail but also won't be too big to be computationally inefficient/cause any image distortion.:

![og](https://github.com/eackerm2/Neural-Networks-Semester-Project/assets/122949257/b59fb42a-3542-45a5-9183-2f2f498e5da9)
![og2](https://github.com/eackerm2/Neural-Networks-Semester-Project/assets/122949257/723168ef-8fca-4f4d-934c-29879aca966c)
![og3](https://github.com/eackerm2/Neural-Networks-Semester-Project/assets/122949257/5e29569e-b56d-40d1-9169-e20d506f76c7)
![og4](https://github.com/eackerm2/Neural-Networks-Semester-Project/assets/122949257/bc59dc70-964a-4d2a-9c80-a82d32980e01)

_Worked Together in Hesburg Library_

# Part 3: First Solution and Validation Accuracy
## SOURCE CODE INSTRUCTIONS
Download the VALIDATION notebook and the Two Validation images. Go into Model.Txt and get the link to the path of trained weights (GitHub wouldn't let us upload bc it was too big). Also to note, inside of the notebook you can insert the two images into the runtime manually by just uploading them on the side bar, but for some reason it doesn't appear like you can do that with the trained weights. So just upload them to drive anywhere and then in the code we have an input line for the path to that. Sorry for any inconvience for that but we don't think theres a way around it.

## Justification of Architecture
### General Architecture:
For our project, we had decided to explore the technique of using a Siamese network to help train our model. The basis is that our network has two inputs so that it can compare two patterns and output a value corresponding to the similarity between them. That way the network could learn the difference between real signatures and fake signatures via training.

(Background on Training)
We hope for the model to be subject-agnostic. To attempt to make this happen, we split the data intentionally so that we broke it up into subjects for the data. There are 30 subjects worth of raw data, and within each subject 5 real signatures and 5 forged signatures. We then applied some mirroring techniques to increase our input data, so there are 1200 total signatures, 20 real and 20 fake for each subject. In the division of our data into training, validation, and testing sets, we made sure that all samples taken from any one individual were in the same data loader. This was in an effort to make sure that the model hadn’t seen any signatures in training that could appear in validation. Had we simply randomly assigned samples to each data set, the model could have seen samples from all subjects in some form during training, and so as a result would not have been subject agnostic due to background influence in some form on all of the samples being used. Simply put, the model could have overfit on the individual writers. 

(Loss Function & Optimizer)
Since we are attempting a Siamese architecture, we are trying to differentiate between the input images. Upon further research, we found that the contrastive loss function may be of use to us since we are trying to determine samples as similar (if they are both genuine or both forged) or dissimilar (one signature is real and the other is fake). Our thought is that you can give the network a real signature of a person, and then a new signature that is either real or fake. Based on the differences observed between the signatures, it should then be able to say the second signature is genuine or forged. The optimizer we used was the Adam optimizer. One of the reasons we decided to use it is because it pretty much only has one hyperparameter that needs to be tuned, which is just the learning rate in our case (other options exist with Adam but we felt we’d start simple).

(Specifics of the Architecture)
Generally speaking, the overarching architecture includes that of a Convolutional Neural Network followed by a fully connected layer. Inside of the Convolutional Network part of our model, we have 4 convolutional layers all with a ReLU activation function and batch normalization and with max pooling implemented in the first 3 layers. The number of convolutional layers was determined as we didn’t want training to take too long by adding too many layers. Consequently, the dimensions of the layers are not overly large either. These are both things that we can go back at the end and fine tune if we see fit. We experimented with some different kernel sizes, but felt 7x7 would be big enough, as anything smaller might not capture larger spatial patterns. The input to the network is 2 images, both 75x200 (standardized in preprocessing). The layers in order move from input and output sizes of 3, 64 to 64, 128 to 128, 128 to 128, 256. The fully connected section of our network then has 2 layers, a linear layer moving from 9216 inputs to 4096 outputs with a ReLU activation function and then dropout for regularization, and a second linear layer moving from 4096 inputs to 1 output. 

### Classification Accuracy
The results of our accuracy of classification are as shows through one of our trains:

![accuracy](https://github.com/eackerm2/Neural-Networks-Semester-Project/assets/124210497/add8a19b-68bf-4e1b-acab-e4fd54f9ff17)


We train over the training data and after each epoch report the validations results. The learning appears to be unstable, as well as it doesn’t seem to be improving between epochs. We believe that the issue lies in our training loop and believe it could be something small. It could also be the format in which we input our data. At this moment, our results are only slightly better than 50/50, so we need to improve drastically. We don’t believe that the overarching architecture with the CNN and FF layers are the issue, rather something dealing with the changing of weights.

### Future Improvements
While we are happy to have code running with no physical errors, there obviously are a few different things we know that we could drastically improve. Our initial thought for training our model was to give it a real signature and a fake signature for the training samples, so that it can learn the difference between the two. We then included some real and real samples along with the real and fake. We believe that potentially training it on more pairs of real and real could help for the generalization of the model. We don’t want it to be overfit to specific outcomes. It would require a lot of data preparation again too, since we’d have to make sure we aren’t repeating data points, and there is enough of the data matched up. We also need to consider how many combinations of real signatures to use. For example for person A, we could pair signature 1 with signatures 2,3,4, and 5 which would result in 4 new data points, but there are also many other real on real combinations possible. If we decide to test this out, we will have to start as soon as possible as cleaning and preparing the data was what took the longest between the last deliverable and this one.

Another item for improvement is potentially exploring different data augmentation techniques rather than just the mirroring in 3 directions other than the original. This could help improve the robustness of the network by providing more data to train with. This may or may not help, as we know that more data doesn’t necessarily mean more accuracy if we are still dealing with the same original data. Again, we want to avoid potential overfitting, so we can explore this and see what happens. We also want to avoid changing the spatial frequency like blurring or sharpening the data, so while data augmentation might be useful we still need to be careful.

We also could see the effect of adding more layers to our initial CNN. We could also adjust and or implement things like batch normalization and dropout and see how that will affect the network’s accuracy. One thing we also didn’t experiment much was changing the learning rate and the optimizer, these things potentially could have an impact on the model’s performance. So tuning the hyperparameters is definitely a must going forward.



### Breakdown of Work:
Both:
 - Worked in Hesburgh Library/LaFun together
 - Went through many iterations to get code running with no errors
 - Attempted to debug issue with learning
 - Worked on data loading and architectual design together
 - Worked on training and evaluation techniques

Patrick:
 - Downloading and getting raw data into Drive
 - Creation of preprocessing functions like flipping and converting to grayscale
 - Original architecture design considerations and code setup
 - Training and evaluation design

Evan:
 - Function to resize all of the images to standard size
 - Preprocessing Data to tensors and prepping train/validation split
 - Created data loader
 - Prepped final report outline going into meeting


