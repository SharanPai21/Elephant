# Import the libraries
import librosa  # for audio processing
import numpy as np  # for numerical operations
import sklearn  # for machine learning
import matplotlib.pyplot as plt  # for visualization

# Load the audio file
audio_file = "e.wav"  # change this to your file name
y, sr = librosa.load(audio_file)  # y is the audio signal, sr is the sampling rate

# Extract the features
mfcc = librosa.feature.mfcc(y, sr)  # mfcc is a matrix of mel-frequency cepstral coefficients
mfcc = mfcc.T  # transpose the matrix

# Check if the mfcc matrix is empty
if mfcc.size == 0:
    print("The mfcc matrix is empty. Please check the audio file.")
    exit()

# Normalize the features
mfcc = sklearn.preprocessing.scale(mfcc, axis=0)

# Define the labels
# Assume that the audio file has two labels: 0 for non-elephant sound, 1 for elephant sound
# You can change the labels according to your data
labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  # change this to your labels
labels = np.array(labels)  # convert to numpy array

# Check if the label array is the same length as the mfcc matrix
if len(labels) != len(mfcc):
    print("The length of the label array is not equal to the length of the mfcc matrix. Please check the labels.")
    exit()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(mfcc, labels, test_size=0.2,
                                                                               random_state=42)

# Train a classifier
# You can use any classifier you want, here we use a logistic regression
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X_train, y_train)  # train the classifier on the train set

# Test the classifier
y_pred = clf.predict(X_test)  # predict the labels on the test set
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)  # compute the accuracy
print("The accuracy of the classifier is:", accuracy)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y, label="Audio signal")  # plot the audio signal
plt.plot(labels * 0.5, label="True labels")  # plot the true labels
plt.plot(clf.predict(mfcc) * 0.5, label="Predicted labels")  # plot the predicted labels
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Elephant trumpet sound detection")
plt.show()
