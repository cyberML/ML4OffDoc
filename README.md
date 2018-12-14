# ML4OffDoc
ML4OffDoc is about using machine learning (ML) for Microsoft (MS) Office documents (Word, Excel and PowerPoint) based malware detection. 
One of the approaches we proposed here is to apply a random forest-based ML model to automate the detection of MS Office document-based cyber-attacks.
In this section, you will be able to download the model source code and run it with your own data feed.
# Get the source
git clone git@github.com:cyberML/ML4OffDoc.git demo

cd demo
# Usage
python rf4doc.py path/to/feed
Note: the feed file needs to be CSV formatted
# Data feed and feature engineering
Due to commercial reasons, demo data feed is not provided, but you can create your own version for the specific type of your application. To get reasonably good results, please follow the steps below:
1.	Extract as many features as possible from MS Office documents (or other file types).  
2.	All feature values need to be numerical. If you have categorical features, please convert them to numeric attributes using one-hot encoding or other methods.
3.	Label malware files with 1s, and benign files with 0s.
4.	Save the data into a csv file with column names: Label, Feature-1,Feature-2, …. 

# Dependencies
Python 3+
Packages: pandas, numpy, sklearn
# Notes
To get optimal results, you will need to tune some parameters such as number of trees, train/validation ratio. 



