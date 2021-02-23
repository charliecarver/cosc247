"""
Note: Changes from D3 are highlighted with "New" comments
"""

from ast import literal_eval
import pandas as pd
import numpy as np
import scipy.sparse
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.feature_extraction.text
import sklearn.model_selection
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.tree
import sklearn.linear_model
import sklearn.linear_model
import sklearn.model_selection
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import statistics
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

"""
Flags
"""

# Set to True to use the Train.csv file and output predictions CSV
useTestCSV = True

# File paths
testPath = 'Test.csv'
trainPath = 'Train.csv'

# NLP params
# New: Hyperparameter optimization
NGRAM_SIZE = 4
COMMON_WORD_THRESHOLD = 10
stemmer = nltk.stem.porter.PorterStemmer()

"""
Processing Funcs
"""

# Preprocess textual data
def preprocessForTextClassification(df):
    df['reviewText'] = df['reviewText'].fillna("")
    df['summary'] = df['summary'].fillna("")
    p = df.groupby('amazon-id').agg({
        'reviewText': ' '.join,
        'summary': ' '.join,
    })
    p['reviewText'] = p['reviewText'] + " " + p['summary']
    return p


# Train text classifier
def trainTextFrequency(df):
    P = preprocessForTextClassification(df)
    # New: Vectorizer change
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,NGRAM_SIZE), min_df=COMMON_WORD_THRESHOLD, preprocessor=lambda token: stemmer.stem(token))
    X1 = vectorizer.fit_transform(P['reviewText'])
    return X1, vectorizer


# Create text matrix for NLP
def getTextMatrix(df, word_indices):
    P = preprocessForTextClassification(df)
    X1 = word_indices.transform(P['reviewText'])
    return X1


# Column normalization
def normalizeColumnData(input_data):
    for feature in input_data:
        input_data[feature] = (input_data[feature] - input_data[feature].min()) / (
                input_data[feature].max() - input_data[feature].min())


# Process numerical data
def processNumerical(df):

    # Drop text data
    df = df.drop(columns=['title', 'categories', 'songs', 'related', 'reviewTime', 'label', 'root-genre'])

    # Process release year
    df['first-release-year'].fillna((df['first-release-year'].median()), inplace=True)
    df['first-release-year'] = df['first-release-year'].apply(lambda x: 1 if x > 1990 else 0)
    df['firstReleaseYear'] = df['first-release-year']
    df.drop(columns='first-release-year', inplace=True)

    # Transform helpful into "ratio" of being helpful
    df['helpful'] = df['helpful'].apply(lambda x: np.nan if literal_eval(x)[1]== 0 else literal_eval(x)[0]/literal_eval(x)[1])
    df['helpful'].fillna((df['helpful'].median()), inplace=True)

    # Review counter for each review (will eventually be summed)
    df['reviewCount'] = 1

    # Return processed data
    return df


"""
Main code
"""

# Print out runtime conditions
if (useTestCSV):
    print("useTestCSV = True, using Test.csv to generate predictions")
else:
    print("useTestCSV = False, using training data to validate model")
print("Training file location: ", trainPath)
print("Testing file location: ", testPath)

# Load dataframes
print("Loading training csv file...")
dfTrain = pd.read_csv(trainPath)
if useTestCSV:
    print("Loading testing csv file...")
    dfTest = pd.read_csv(testPath)

# Train text classifier on training data
print("Training text classifier...")
trainingTextMatrix, wordIndices = trainTextFrequency(dfTrain)

# Process textual data
if useTestCSV:
    print("Getting text matrix for testing data...")
    testTextMatrix = getTextMatrix(dfTest, wordIndices)

# Process numerical data
print("Processing numerical data in training file...")
dfTrain = processNumerical(dfTrain)
if useTestCSV:
    print("Processing numerical data in testing file...")
    dfTest = processNumerical(dfTest)

# Aggregate training data and normalize
print("Grouping training data by amazon-id and aggregating...")
# New: Added additional numerical fields
trainData = dfTrain.groupby('amazon-id').agg({
    'unixReviewTime': 'mean',
    'price': 'mean',
    'overall': lambda x: 1 if np.mean(x) > 4.5 else 0,
    'salesRank': 'mean',
    'helpful': 'mean',
    'firstReleaseYear': 'mean',
    'reviewCount': 'sum'
})
normalizeColumnData(trainData)

# Split data into dependent/independent vars
if useTestCSV:

    # Aggregate testing data from CSV file
    # New: Added additional numerical fields
    print("Grouping testing data by amazon-id and aggregating...")
    testData = dfTest.groupby('amazon-id').agg({
        'unixReviewTime': 'mean',
        'price': 'mean',
        'salesRank': 'mean',
        'helpful': 'mean',
        'firstReleaseYear': 'mean',
        'reviewCount': 'sum'
    })
    normalizeColumnData(testData)

    # Split data
    print("Splitting testing/training data into dependent and indepedent variables...")
    
    # New: Ablation columns + KBest
    # Construct X training data
    ablationColumns = ['firstReleaseYear', 'reviewCount']
    XTrain = scipy.sparse.csr_matrix(scipy.sparse.hstack(
        (trainingTextMatrix, scipy.sparse.csr_matrix(trainData[ablationColumns].to_numpy()))
    ))

    # Construct y training data
    yTrain = trainData['overall'].to_numpy()
    
    # Select KBest features from training set
    selection = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=7000).fit(XTrain, yTrain)
    
    # Transform X training data based on KBest features 
    XTrain = selection.transform(XTrain)
    
    # Construct testing data
    XTest = scipy.sparse.csr_matrix(scipy.sparse.hstack(
        (testTextMatrix, scipy.sparse.csr_matrix(testData[ablationColumns].to_numpy()))
    ))

    # Transform X testing data based on KBest features 
    XTest = selection.transform(XTest)

else:

    # If we're just testing our classifier, split the training data into training + testing datasets
    # Construct X training data
    # New: Ablation testing + KBest
    print("Splitting training data into testing and training sets...")
    ablationColumns = ['firstReleaseYear', 'reviewCount']
    XTrain = scipy.sparse.csr_matrix(scipy.sparse.hstack(
        (trainingTextMatrix, scipy.sparse.csr_matrix(trainData[ablationColumns].to_numpy()))
    ))
    
    # Construct y training data
    yTrain = trainData['overall'].to_numpy()
    
    # Select KBest features from training set
    selection = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=7000).fit(XTrain, yTrain)
    
    # Transform X training data based on KBest features 
    XTrain = selection.transform(XTrain)
    
    # Split training data into training + testing sets
    XTrain, XTest, yTrain, yTest = sklearn.model_selection.train_test_split(XTrain, trainData['overall'].to_numpy(), test_size=0.3, shuffle=True)

# Testing model
if not useTestCSV:
    print("Testing model with 10-fold cross-validation...")
    kf = sklearn.model_selection.KFold(n_splits=10, shuffle=True)
    f1s = []
    for trainIdx, testIdx in kf.split(XTrain):
        xTrainKF, xTestKF = XTrain[trainIdx], XTrain[testIdx]
        yTrainKF, yTestKF = yTrain[trainIdx], yTrain[testIdx]
        # New: Hyperparameter optimization for LR
        clf = sklearn.linear_model.LogisticRegression(max_iter=100000, multi_class='multinomial', C=6.97, class_weight='balanced')
        clt = clf.fit(xTrainKF, yTrainKF)
        f1 = sklearn.metrics.f1_score(yTestKF, clt.predict(xTestKF), average='weighted')
        print("\tF1 {}".format(f1))
        f1s.append(f1)
    # New: Added more stats
    print("Mean F1: ", statistics.mean(f1s))
    print("STDV F1: ", statistics.stdev(f1s))
    print("RSD F1: ", statistics.stdev(f1s)/statistics.mean(f1s)*100, "%")

# Output CSV file with predictions
if useTestCSV:
    print("Training model...")
    # New: Hyperparameter optimization for LR
    clf = sklearn.linear_model.LogisticRegression(max_iter=100000, multi_class='multinomial', C=6.97, class_weight='balanced')
    clt = clf.fit(XTrain, yTrain)
    yPreds = clt.predict(XTest)
    output = pd.DataFrame({'amazon-id': testData.index, 'Awesome': yPreds})
    output.to_csv('./Product_Predictions.csv', index=False)
    print("Output predictions to './Product_Predictions.csv'")

print("Done!")
