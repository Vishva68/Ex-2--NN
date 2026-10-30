<H3>Name: VISHVANANDH N</H3>
<H3>Register no: 212224240186</H3>
<H3>Date 30-10-2026 </H3>
<H3>Experiment No. 2 </H3>
<h3> Implementation of Perceptron for Binary Classification</h3>

# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:

STEP 1: Import the required Python libraries such as NumPy, Pandas, Matplotlib, and Scikit-learn.
Define the Perceptron class with initialization, training, and prediction methods.

STEP 2: Load the Iris dataset from the UCI repository.
Display the first few records to verify the dataset.

STEP 3: Extract the input features and output class labels.
Visualize the complete dataset using a 3D scatter plot.

STEP 4: Select the first two classes and two input features.
Plot them to verify that the data is linearly separable.

STEP 5: Convert the class labels into binary values (+1 and −1).
Apply feature scaling to normalize the input features.

STEP 6: Split the dataset into training and testing sets.
Initialize the Perceptron with learning rate, weights, and bias.

STEP 7: Train the Perceptron for a fixed number of epochs.
Compute the net input and predict the output for each sample.

STEP 8: Compare the predicted output with the actual label.
Update the weights and bias using the Perceptron learning rule.

STEP 9: Count the number of misclassified samples.
Store the error value after each training epoch.

STEP 10: Test the trained Perceptron using the test dataset.
Predict the class labels for unseen samples.

STEP 11: Calculate the classification accuracy.
Evaluate the overall performance of the model.

STEP 12: Plot the misclassification errors against epochs.
Analyze the learning performance of the Perceptron model.

# PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
  def __init__(self,learning_rate=0.1):
    self.learning_rate = learning_rate
    self._b = 0.0
    self._w = None
    self.misclassified_samples = []
  def fit(self, x: np.array, y: np.array, n_iter=10):
    self._b = 0.0
    self._w = np.zeros(x.shape[1])
    self.misclassified_samples = []
    for _ in range(n_iter):
      errors = 0
      for xi, yi in zip(x, y):
        update = self.learning_rate * (yi - self.predict(xi))
        self._b += update
        self._w += update * xi
        errors += int(update != 0.0)
      self.misclassified_samples.append(errors)

  def f(self, x: np.array) -> float:
      return np.dot(x, self._w) + self._b

  def predict(self, x: np.array):
      return np.where(self.f(x) >= 0, 1, -1)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print(df.head())

y = df.iloc[:, 4].values
x = df.iloc[:, 0:3].values

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Iris data set')
ax.set_xlabel("Sepal length in width (cm)")
ax.set_ylabel("Sepal width in width (cm)")
ax.set_zlabel("Petal length in width (cm)")

ax.scatter(x[:50, 0], x[:50, 1], x[:50, 2], color='red',
         marker='o', s=4, edgecolor='red', label="Iris Setosa")
ax.scatter(x[50:100, 0], x[50:100, 1], x[50:100, 2], color='blue',
         marker='^', s=4, edgecolor='blue', label="Iris Versicolour")
ax.scatter(x[100:150, 0], x[100:150, 1], x[100:150, 2], color='green',
         marker='x', s=4, edgecolor='green', label="Iris Virginica")
plt.legend(loc='upper left')
plt.show()

x = x[0:100, 0:2] 
y = y[0:100]

plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x',
          label='Versicolour')
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
plt.show()

y = np.where(y == 'Iris-setosa', 1, -1)
x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)

classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)
print("accuracy", accuracy_score(classifier.predict(x_test), y_test)*100)

plt.plot(range(1, len(classifier.misclassified_samples) + 1),classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()

 ```

# OUTPUT:

<img width="344" height="127" alt="image" src="https://github.com/user-attachments/assets/de80ef3f-d4d9-4974-ba3d-9f52b097d50e" />


<img width="450" height="486" alt="image" src="https://github.com/user-attachments/assets/618f9e20-84a0-41e4-bcda-7e30e093585b" />


<img width="640" height="490" alt="image" src="https://github.com/user-attachments/assets/f25583b9-3219-4546-b5fc-fc47971a3b37" />


<img width="621" height="515" alt="image" src="https://github.com/user-attachments/assets/1a8e3f6b-bf6b-481b-8fea-72fdaa0f3814" />


# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
