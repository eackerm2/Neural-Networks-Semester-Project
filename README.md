# Neural-Networks-Semester-Project
### Evan Ackerman and Patrick Schlosser
This is the repository for our semester project in CSE 40868.

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

### Breakdown of Work:
#### Patrick:
#### - Downloading and getting raw data into Drive
#### - Creation of preprocessing functions like flipping and converting to grayscale

#### Evan:
#### - Function to resize all of the images to standard size
#### - Preprocessing Data to tensors and prepping train/validation split
