# NYC_Traffic_Safety_Project
I tidied and reorganized the codes of the first project I did (as a team with my classmates). This repo includes the original data of maps and accidents in NYC, also Python codes to do preprocessing and feed them into a CNN model.

## Background
Traffic accidents are becoming more and more common causes of injuries and deaths in cities. Only in New York, on average 622 traffic accidents happened everyday in the past days in 2018, according to NYC Open Data. <br>
The ability to predict future accidents (e.g., where, when, or how) is thus very useful not only to public safety stakeholders (e.g., police, autonomous firms) but also transportation administrators and individual travelers. <br>
Many scholars have conducted a lot research on causes of traffic accidents but little of them focuses on the road design. However, there are many cases that a terribly-designed crossroad causing accidents every year. <br>

## What we did
We trained a Convolutional Neural Network (CNN) using satellite images of crossroads as features and numbers of nearby traffic accidents as labels to predict road safety level.

## Our data
**Satellite images** from Google API: 640*640*3 RGB images. <br>
First, we computed the coordinates of every intersections in NY, Manhattan, then use these coordinates as centers to generate satellite images. We got more than 8k images of intersections in Manhattan.

**NY Traffic Accident Data** from NYPD: contains coordinates of every traffic accident in NY from 2012 to 2018. <br>
Based on the center coordinates we computed, we counted the number of accidents nearby (in the image). We then transferred these numbers into 5 categories to label the safety of these intersections.

### Data Visualization


## Our model
We will use a structure with two convolutional layers followed by max pooling and a flattening out of the network to fully connected layers to make predictions.
Our baseline network structure can be summarized as follows:
1.	Convolutional input layer, 32 feature maps with a size of 3×3, a rectifier activation function and a weight constraint of max norm set to 3.
2.	Dropout set to 20%.
3.	Convolutional layer, 32 feature maps with a size of 3×3, a rectifier activation function and a weight constraint of max norm set to 3.
4.	Max Pool layer with size 2×2.
5.	Flatten layer.
6.	Fully connected layer with 512 units and a rectifier activation function.
7.	Dropout set to 50%.
8.	Fully connected output layer with 10 units and a softmax activation function.
A logarithmic loss function is used with the stochastic gradient descent optimization algorithm configured with a large momentum and weight decay start with a learning rate of 0.01.

## Results
Training Accuracy: 75.68% <br>
Test Accuracy: 74.29% <br>
Our model has a good performance, with a probability of about 75% to correctly label a new satellite image with its safety level. 
We also designed an API that is able to predict the safety level when feeding with a coordinate or road image.

## Project revisiting after two years
This was the first project I did (as a team) after learning about machine learning and neural networks. It might look naïve now and the product has little practical usage. Since CNN is a black box, even we can predict the road safety, we still need to look into the details of these road to figure out why it is safe or dangerous. 

However, this project does provide an interesting perspective to look into this Civil Engineering problem. The project could be improved from two stand points: <br>
1. Model: We could turn back to more explainable models from traditional machine learning, such as tree models and logistic regression.
2 .Feature: We could gather features from more dimensions beyond images. Examples include heights of the surrounding buildings, angels of the crossroads and number of lanes. While we may not have off-the-shelf data about the angels and lane numbers, satellite images do include these information and this is where a Convolutional Neural Network may help.
