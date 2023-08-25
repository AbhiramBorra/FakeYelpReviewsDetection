# FakeAmazonReviewsDetection
Use ti-idf and DBSACAN to perform data processing
We will be using TF-IDF on our text feutures. The TF-IDF metric will scan thorugh text and calculate the frequency of appereance of words within the text. After getting frequency, well plot them and use DBSCAN to cluster and classify each cluster. Our feuture for "rating" is straightfoward, therefore we won't do any preprocessing. For our "recommended" feuture, well convert into binary, 1 for 'yes' and 0 for 'No'.
