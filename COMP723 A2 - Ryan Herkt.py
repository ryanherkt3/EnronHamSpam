# COMP723 Assignment 2
# Name: Ryan Herkt
# ID: 18022861

import glob
import numpy as np
import re
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, \
    precision_score, recall_score, f1_score

# Base path where enron folders are stored
base_path = r"C:\AUT 2021\COMP723 - DMKE\Assignment 2" # change for your machine x

''' DATA PREPROCESSING FUNCTION '''
# Tokenization, stemming, making words lowercase
def data_preprocessing(X, train_set):
    emails = []

    for sen in range(0, len(X)):
        # Remove all the special characters
        email = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        email = re.sub(r'\s+[a-zA-Z]\s+', ' ', email)

        # Remove single characters from the start
        email = re.sub(r'\^[a-zA-Z]\s+', ' ', email)

        # Substituting multiple spaces with single space
        email = re.sub(r'\s+', ' ', email, flags=re.I)

        # Only remove prefixed 'b' if pre-processing training set 2
        if (train_set == 2):
            # Removing prefixed 'b'
            email = re.sub(r'^b\s+', '', email)

        # Converting to Lowercase
        email = email.lower()

        # Lemmatization
        email = email.split()

        from nltk.stem import WordNetLemmatizer

        stemmer = WordNetLemmatizer()
        email = [stemmer.lemmatize(word) for word in email]
        email = ' '.join(email)

        emails.append(email)

    return emails
    
''' CLASSIFIER ALGORITHM OUTPUT FUNCTION '''
def classifier_results(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, np.ravel(y_train,order='C'))

    '''Now predict on the testing data'''
    y_pred = classifier.predict(X_test)

    '''Print the evaluation metrices'''
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def training_set_one():
    ''' Get email files from folders - training set one '''
    # Subfolders
    enrons = ["\enron1","\enron2","\enron3","\enron4","\enron5"]
    type = ["\ham", "\spam"]
    
    '''
    Training set one: Split into 70% training set and 30% test set, and maintain 
    the ham:spam ratio. 
    '''
    # Create 10 lists to store the enron files in
    # e.g (files[0] = enron1/ham, files[1] = enron1/spam, files[2] = enron2/ham, etc.)
    files = [list() for x in range(10)]

    # Store enron emails filepaths in the lists
    count = 0
    for e in enrons:    # loop through the five enron folders
        for t in type:  # loop though the ham and spam folders
            files[count] = glob.glob(base_path + e + t + "\*.txt")
            count += 1

    # Number of files in corpus = 27716
    # 70% of files in corpus = 19401
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Function which reads the content from the text files, and adds it
    # to training and test sets while maintaining the ham:spam ratio, 
    # prior to preprocessing the data.
    # It also ignores non-ASCII characters
    def my_train_test_split(ham_files, spam_files, num_ham_train, num_spam_train):
        #add files and values to training set
        for x in range(0, num_ham_train):
            X_train.append(open(ham_files[x], 'r', encoding='ascii', errors='ignore').read())
            y_train.append(0)
        for x in range(0, num_spam_train):
            X_train.append(open(spam_files[x], 'r', encoding='ascii', errors='ignore').read())
            y_train.append(1)
        
        #add rest of the files and values to test set
        for x in range(num_ham_train, len(ham_files)):
            X_test.append(open(ham_files[x], 'r', encoding='ascii', errors='ignore').read())
            y_test.append(0)
        for x in range(num_spam_train, len(spam_files)):
            X_test.append(open(spam_files[x], 'r', encoding='ascii', errors='ignore').read())
            y_test.append(1)

    # Ham/Spam Ratios: enron1, enron2, enron3 = 3:1 (approx)
    # Ham/Spam Ratios: enron4, enron5 = 1:3 (approx)
    # A function which determines how to maintain the ham:spam ratio
    # when splitting the 70% training and 30% testing sets, based on the number
    # of files in the enron{x}/ham and enron{x}/spam folders.
    #
    # If H:S ratio is in favour of the ham files (i.e. 3:1):
    # No. of ham files in training set = [number of ham+spam files * 0.7]/[[(# spam files)/(# ham files)] + 1]
    # No. of spam files in training set = [number of ham+spam files * 0.7] - No. of ham files in training set
    #
    # If H:S ratio is in favour of the spam files (i.e. 1:3):
    # No. of spam files in training set = [number of ham+spam files * 0.7]/[[(# spam files)/(# ham files)] + 1]
    # No. of ham files in training set = [number of ham+spam files * 0.7] - No. of ham files in training set
    def t_t_splits(ham_files, spam_files):
        total = len(ham_files) + len(spam_files)

        if(len(ham_files) > len(spam_files)): # ham:spam ratio 3:1
            no_ham = round((total*0.7)/((len(spam_files) / len(ham_files))+1))
            no_spam = round((total * 0.7) - no_ham)
        else: # ham:spam ratio 1:3
            no_spam = round((total*0.7)/((len(ham_files) / len(spam_files))+1))
            no_ham = round((total*0.7) - no_spam)
        my_train_test_split(ham_files, spam_files, no_ham, no_spam)

    print('splitting data')
    t_t_splits(files[0], files[1]) # train/test split enron 1 data
    t_t_splits(files[2], files[3]) # train/test split enron 2 data
    t_t_splits(files[4], files[5]) # train/test split enron 3 data
    t_t_splits(files[6], files[7]) # train/test split enron 4 data
    t_t_splits(files[8], files[9]) # train/test split enron 5 data

    # convert y_train and y_test to numpy array here:
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    '''DATA PREPROCESSING FOR TRAINING SET 1'''
    print('preprocessing data')
    X_train = data_preprocessing(X_train, 1)
    X_test = data_preprocessing(X_test, 1)

    ''' CLASSIFY TEXTS FROM TRAINING SET 1 '''
    # Convert the word to a vector using TF-IDF. Also removes stopwords
    vectorizer = CountVectorizer(max_features=1000, min_df=8, max_df=0.25, stop_words=stopwords.words('english'))
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.fit_transform(X_test).toarray()

    tfidfconverter = TfidfTransformer()
    X_train = tfidfconverter.fit_transform(X_train).toarray()
    X_test = tfidfconverter.fit_transform(X_test).toarray()
    
    # Classifier 1 (Naive Bayes)
    print("\nTraining set 1 - Multinomial NB")
    classifier_results(MultinomialNB(), X_train, X_test, y_train, y_test)

    # Classifier 2 (Neural Network)
    print("\nTraining set 1 - Neural Network")
    # get same results every time w/ random_state
    classifier_results(MLPClassifier(random_state=0), X_train, X_test, y_train, y_test)

    # Classifier 3 (My choice - SVM)
    print("\nTraining set 1 - SVM")
    classifier_results(LinearSVC(), X_train, X_test, y_train, y_test)

def training_set_two():
    ''' Get email files from folders - training set two '''
    print("enron 1")
    enron1_files = load_files(base_path + "\enron1")
    print("enron 2")
    enron2_files = load_files(base_path + "\enron2")
    print("enron 3")
    enron3_files = load_files(base_path + "\enron3")
    print("enron 4")
    enron4_files = load_files(base_path + "\enron4")
    print("enron 5")
    enron5_files = load_files(base_path + "\enron5")
    
    '''
    Training set two: Use enron1, enron3, enron5 for training; and use enron2, 
    enron4 for testing. 
    '''
    # Manually set training and test sets rather than use train_test_split
    X_train = (enron1_files.data + enron3_files.data + enron5_files.data)
    X_test = (enron2_files.data + enron4_files.data)
    y_train = np.concatenate((enron1_files.target, enron3_files.target, enron5_files.target), axis = None)
    y_test = np.concatenate((enron2_files.target, enron4_files.target), axis = None)

    '''DATA PREPROCESSING FOR TRAINING SET 2'''
    print('preprocessing data')
    X_train = data_preprocessing(X_train, 2)
    X_test = data_preprocessing(X_test, 2)

    # Convert the word to a vector using TF-IDF. Also removes stopwords
    vectorizer = CountVectorizer(max_features=1000, min_df=8, max_df=0.25, stop_words=stopwords.words('english'))
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.fit_transform(X_test).toarray()
    tfidfconverter = TfidfTransformer()
    X_train = tfidfconverter.fit_transform(X_train).toarray()
    X_test = tfidfconverter.fit_transform(X_test).toarray()
    
    ''' CLASSIFY TEXTS FROM TRAINING SET 2 '''
    # Classifier 1 (Naive Bayes)
    print("\nTraining set 2 - Multinomial NB")
    classifier_results(MultinomialNB(), X_train, X_test, y_train, y_test)

    # Classifier 2 (Neural Network)
    print("\nTraining set 2 - Neural Network")
    # get same results every time w/ random_state
    classifier_results(MLPClassifier(random_state=0), X_train, X_test, y_train, y_test)

    # Classifier 3 (My choice - SVM)
    print("\nTraining set 2 - SVM")
    classifier_results(LinearSVC(), X_train, X_test, y_train, y_test)
    
print('Testing classification algorithms on training set one')
training_set_one()
print('Testing classification algorithms on training set two')
training_set_two()