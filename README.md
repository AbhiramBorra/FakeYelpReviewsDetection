# FakeYelpReviewsDetection
We are changing the project dataset from Amazon reviews to Yelp reviews. The goal of our project is still the same to detect fake reviews. The YelpReviews.ipynb file has our latest work.                                                                                        
Dataset:
https://www.kaggle.com/datasets/abidmeeraj/yelp-labelled-dataset
The dataset provided by Abidmeeraj on Kaggle is a list of over 35,000 yelp consumer reviews about hotels located in North America. 


Preprocessing : 
Text to numerical conversion - For the preprocessing stage we are using TF-IDF to process the text into numbers.
Finally, we ran the TF-IDF metric, which scans through text and calculates the frequency of appearance of words within the text during the TF stage and measures global importance during the IDF stage. 

Data Cleaning - During the preprocessing stage, we dropped any data that didn't add any value, and cleared the rest of the data by removing any null entries.

Feature Engineering - We expanded characteristics through creating novel features from those that already exist, primarily through modifying data using polynomial, logarithmic, or feature multiplication operations.


Model Builing:
For our initial model we will run neural network based algorithm on 3 dimensions with the major goal to identify the real and fake reviews cluster within the dataset.


Evaluation:
We evaluate the performance of our model using accuracy and other metrics by employing the subsequent stages:

Training versus Test Error Comparison - In order to determine the model's performance and accuracy capabilities, we analyze the training and test errors.

Model Fitting Analysis - In order to determine insights into its bias and variance, we examine the models's positioning relative to the underfitting/overfitting graph. 


Features : 
Our feature for "rating" is straightforward, therefore we won't do any preprocessing. 
However, for our "recommended" feature, for simplification of rpresentation and analysis we have converted it into binary, 1 for 'yes' and 0 for 'No'.

First model evaluate:
Epoch 48/50
71/71 [==============================] - 2s 31ms/step - loss: 0.2151 - accuracy: 0.9278
Epoch 49/50
71/71 [==============================] - 3s 41ms/step - loss: 0.1917 - accuracy: 0.9363
Epoch 50/50
71/71 [==============================] - 2s 28ms/step - loss: 0.2292 - accuracy: 0.9189

Training MSE: 0.07677777777777778                                                                                                                                                                                                    
Testing MSE: 0.1                                                                                                                                                                                             
From the MSE data above, we think the first model is a bit overfit because the testing MSE is 25% higher than the training MSE, the training error is lower.


Result :


Conclusion : 
Our research seeks to offer an approach for detecting false Yelp reviews. We anticipate that our attempts will contribute to strengthen customer trust in hotel management. 


Contributions :  
Group members -                                                                                                                                                    
Abhi Borra,	aborra@ucdavis.edu                                                                                                            
Rohit Singh,	rkssingh@ucdavis.edu                                                                                                              
Jaryd Bones,	jbones@ucdavis.edu                                                                                                                 
Jose Navarro,	janavarro@ucdavis.edu                                                                                                             
Zhixuan Qiu,	zxqiu@ucdavis.edu



