# FakeAmazonReviewsDetection
Dataset: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/code?select=1429_1.csv

The dataset provided by Datafinity on Kaggle is a list of over 34,000 consumer reviews for Amazon products.
We are using TF-IDF to process the text into numbers. Next, we will run DBSACAN on 3 dimensions to identify the real and fake reviews cluster.
During the preprocessing stage, we dropped any data that didn't add any value, and cleared the rest of the data by removing any null entries.
Finally, we ran the TF-IDF metric, which scans through text and calculates the frequency of appearance of words within the text during the TF stage and measures global importance during the IDF stage. 

Group members:                                                                                                                                                    
Abhi Borra,	aborra@ucdavis.edu                                                                                                            
Rohit Singh,	rkssingh@ucdavis.edu                                                                                                              
Jaryd Bones,	jbones@ucdavis.edu                                                                                                                        
Jose Navarro,	janavarro@ucdavis.edu                                                                                                             
Zhixuan Qiu,	zxqiu@ucdavis.edu
