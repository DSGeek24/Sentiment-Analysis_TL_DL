Sentiment analysis is one of the most popular NLP tasks which is extremely useful in gaining an overview of public opinion on certain products, services or topics. Understanding the consumer attitudes and reacting accordingly can be very advantageous for any market research or customer centric approach. Being able to understand the sentiments of users can help in strategizing and planning for future, also a shift in sentiment has been shown to correlate with a shift in stock market.  

In this project, I have taken up three datasets ranging from small, medium and large(in terms of number of instances) and implemented deep learning architectures of Convolutional Neural Networks, LSTM’s and a combination of them with words as the features to predict the sentiment for unseen user reviews(Movie Reviews, Restaurant reviews etc) after learning from sentiment labelled sentences datasets. I compared the performance of each model on different range of datasets using performance metrics like precision, recall, F1 score and accuracy. Two datasets contain binary polarity whereas the one taken from Kaggle deals with multiple polarity(5- negative, somewhat negative, neutral, somewhat positive, positive).   

I performed transfer learning on this NLP task by making use of pre-trained word embedding models made available by Stanford (GloVe). 

Run instructions: 

1. Open the command prompt and get to the path of the given folder.
2. Mention the .tsv file you want to run(among the three datasets) in the .py file  
2. Use command python SentimentAnalysis.py to execute the file. 
3. Once the program is run, accuracy, loss plots and confusion matrix are shown. The classification report along with accuracy is printed. 


Send me an email(dkonreddy@uh.edu) if you want a detailed report about findings and results. 