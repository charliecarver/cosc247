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
NGRAM_SIZE = 2
COMMON_WORD_THRESHOLD = 10
stemmer = nltk.stem.porter.PorterStemmer()

"""
Processing Funcs
"""


# Preprocess textual data
# Todo: Merge with getTextMatrix()?
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
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, NGRAM_SIZE))
    X1 = vectorizer.fit_transform(P['reviewText'])
    return X1, vectorizer


# Create text matrix for NLP
def getTextMatrix(df, word_indices):
    P = preprocessForTextClassification(df)
    X1 = word_indices.transform(P['reviewText'])
    return X1


# Column normalization
# Todo: Merge with processNumerical()
def normalizeColumnData(input_data):
    for feature in input_data:
        input_data[feature] = (input_data[feature] - input_data[feature].min()) / (
                input_data[feature].max() - input_data[feature].min())


# Process numerical data
# Todo: Consider processing additional columns
def processNumerical(df):
    # Drop all textual data
    df = df.drop(columns=['title', 'categories', 'songs', 'related', 'reviewTime'])

    # Drop columns that we need more time to consider processing
    df = df.drop(columns=['label', 'first-release-year', 'root-genre'])
    # df = df.join(pd.get_dummies(df['root-genre']))
    # df = df.drop(columns=['root-genre'])

    # Transform helpful into "ratio" of being helpful
    df['helpful'] = df['helpful'].apply(
        lambda x: np.nan if literal_eval(x)[1] == 0 else literal_eval(x)[0] / literal_eval(x)[1])
    df['helpful'].fillna((df['helpful'].median()), inplace=True)

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
trainData = dfTrain.groupby('amazon-id').agg({
    'unixReviewTime': 'mean',
    'price': 'mean',
    'overall': lambda x: 1 if np.mean(x) > 4.5 else 0,
    'salesRank': 'mean',
    'helpful': 'mean',
})
normalizeColumnData(trainData)

# Split data into dependent/independent vars
if useTestCSV:

    # Aggregate testing data from CSV file
    print("Grouping testing data by amazon-id and aggregating...")
    testData = dfTest.groupby('amazon-id').agg({
        'unixReviewTime': 'mean',
        'price': 'mean',
        'salesRank': 'mean',
        'helpful': 'mean',
    })
    normalizeColumnData(testData)

    # Split data
    print("Splitting testing/training data into dependent and indepedent variables...")
    ytrain = trainData['overall'].to_numpy()
    Xtrain = scipy.sparse.hstack(
        (trainingTextMatrix, scipy.sparse.csr_matrix(trainData.drop(columns='overall').to_numpy()))
    )
    Xtrain = scipy.sparse.csr_matrix(Xtrain)
    testIndex = testData.index
    Xtest = scipy.sparse.hstack(
        (testTextMatrix, scipy.sparse.csr_matrix(testData.to_numpy()))
    )
    Xtest = scipy.sparse.csr_matrix(Xtest)
else:

    # If we're just testing our classifier, split the training data into training + testing datasets
    print("Splitting training data into testing and training sets...")
    Xtrain = scipy.sparse.csr_matrix(scipy.sparse.hstack(
        (trainingTextMatrix, scipy.sparse.csr_matrix(trainData['helpful'].to_numpy().reshape(-1, 1)))
    ))
    ytrain = trainData['overall'].to_numpy()
    Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(Xtrain, trainData['overall'].to_numpy(),
                                                                            test_size=0.3, shuffle=True)

# Testing model
if not useTestCSV:
    print("Testing model with 10-fold cross-validation...")
    kf = sklearn.model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
    f1_vals = []
    for train_index, test_index in kf.split(Xtrain):
        x_train, x_test = Xtrain[train_index], Xtrain[test_index]
        y_train, y_test = ytrain[train_index], ytrain[test_index]
        clf = sklearn.linear_model.LogisticRegression(max_iter=100000, class_weight='balanced')
        clt = clf.fit(x_train, y_train)
        f1 = sklearn.metrics.f1_score(y_test, clt.predict(x_test), average='weighted')
        print("\tF1 {}".format(f1))
        f1_vals.append(f1)
    print("Mean F1: ", statistics.mean(f1_vals))

# Output CSV file with predictions
if useTestCSV:
    print("Training model...")
    lr = sklearn.linear_model.LogisticRegression(max_iter=100000, class_weight='balanced')
    lrTrained = lr.fit(Xtrain, ytrain)
    ypreds = lrTrained.predict(Xtest)
    output = pd.DataFrame({'amazon-id': testIndex, 'Awesome': ypreds})
    output.to_csv('./Product_Predictions.csv')
    print("Output predictions to './Product_Predictions.csv'")

print("Done!")