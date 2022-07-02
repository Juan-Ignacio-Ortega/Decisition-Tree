# DecisitionTree

## 1 - Code and Description
### Code
https://github.com/Juan-Ignacio-Ortega/DecisitionTree/blob/main/DecisionTreeCode_JIOG.py

### Description
https://github.com/Juan-Ignacio-Ortega/DecisitionTree/blob/main/DecisionTreeDescription_JIOG.ipynb

## 2 - Introduction
Inductive inference is the process of going from concrete examples to general models. In one form, the goal is to learn to classify objects or situations by analyzing a set of instances whose classes are known [6].

Data mining techniques basically use the ID3 algorithm since it is the basic classification algorithm. Milija Suknovic (et.al) identified some reusable components from the ID3 algorithm. They propose a generic decision tree framework that supports the design of reusable components. The proposed generic decision tree framework consists of several subproblems that were recognized through analysis of well-known decision tree induction algorithms, namely ID3, C4.5, CART, CHAID, QUEST, GUIDE, CRUISE, and CTREE [ two].

A decision tree is a formalism to express such mappings. A tree is a leaf node labeled with a class or a structure consisting of a test node linked to two or more subtrees. A test node computes some result based on the attribute values ​​of an instance, where each possible result is associated with one of the subtrees [6].

In this work, an algorithm based on ID3 is proposed to generate a decision tree, which is tested using a database to classify the types of drugs that a patient needs, in addition, the results are tested through the performance of their classification.

## 3 - Theoretical framework
### 3.1 Decision trees
3.1.1 Definition
Decision trees (DTs) are a nonparametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules deduced from data features. A tree can be viewed as a piecewise constant approximation [1].

3.1.2 Advantages and Disadvantages
Some advantages of decision trees are:

- Easy to understand and interpret.
Trees can be visualized.

- Requires little data preparation.
Other techniques often require data normalization, creating dummy variables and removing blank values. However, please note that this module does not support missing values.

- The cost of using the tree (ie predicting data) is logarithmic in the number of data points used to train the tree.

- Able to handle numerical and categorical data.
However, the scikitlearn implementation does not support categorical variables at this time. Other techniques are usually specialized in analyzing data sets that have a single type of variable.

- Able to handle multiple output problems.

- Use a white box model. If a given situation is observable in a model, the explanation of the condition is easily explained by Boolean logic. In contrast, in a black box model (for example, in an artificial neural network), the results can be more difficult to interpret.

- Possibility of validating a model through statistical tests. This allows us to account for the reliability of the model.

- It performs well even if its assumptions are somewhat violated by the true model from which the data was generated.

Disadvantages of decision trees include:
- Decision tree learners may create overly complex trees that do not generalize data well. This is called overfitting. To avoid this problem, mechanisms such as pruning, setting the minimum number of samples required at a leaf node, or setting the maximum depth of the tree are necessary.

- Decision trees can be unstable because small variations in the data can generate a completely different tree. This problem is mitigated by the use of decision trees within an ensemble.

- The predictions of the decision trees are not uniform or continuous, but constant approximations by parts, as seen in the previous figure. Therefore, they are not good for extrapolation.

- It is known that the problem of learning an optimal decision tree is NP-complete under various optimization aspects and even for simple concepts. Consequently, practical decision tree learning algorithms are based on heuristic algorithms, such as the greedy algorithm, in which locally optimal decisions are made at each node. Such algorithms cannot guarantee the return of the globally optimal decision tree. This can be mitigated by training multiple trees on an ensemble learner, where features and samples are randomly sampled with replacement.

- There are concepts that are difficult to learn because decision trees do not express them easily, such as XOR, parity problems or multiplexer.

- Decision tree learners create biased trees if they master some classes. Therefore, it is recommended to balance the data set before fitting it to the decision tree [1].

3.1.3 Step by step calculations:
Step 1 - Calculate the entropy of the attributes and choose the largest as root.
Step 2 - Calculate the gain of the remaining attributes and find which attribute is the next node.
Step 3 - Find a node for each root branch if separate classifications on all branches have not yet been arrived at.
Step 4 - Follow the node selection process until the attributes are finished or there are separate classifications in each branch [2].

### 3.2 K-Fold
K-Fold, also called cross-validation, is a procedure that consists of dividing the data into k times and performing tests with each k-partition.
The parameter K indicates how many times the data has to be partitioned. The most used K are 3, 5 and 10 partitions. The K is usually replaced by the number of k-partitions, '10-Fold'.
The method is very popular because the results are less biased and a less optimistic estimate of the performance and accuracy of an algorithm is made.

The general algorithm is as follows:
1. The data is randomly scrambled.
2. The data is separated into k groups. In this case, it is recommended to create copies of the data, each with its corresponding partition. For example, if you want to do a '5-Fold', make five copies of the data. In the first copy, the first fifth part is removed, in the second copy, the second fifth part and so on until the five copies are completed.
3. The parts that were removed will be the test stands. The first test is done with the model from the first copy and the test data that was removed from it, and so on.
4. Quality metrics (error, classi cation rate, accuracy, precision, sensitivity, and F-beta score) are saved.
5. The model and the test bench with which it was made are discarded.
6. Switch to the next model with the next test bench, repeat the procedure from step four until all partitions are tested.

The data is divided as homogeneously as possible, that is, each partition contains approximately the same amount of data. As a graphic way of representing this, we can show the example of the following figure [5]:

![alt text](https://github.com/Juan-Ignacio-Ortega/DecisitionTree/blob/main/KFold.jpeg?raw=true)

Figure 1. Example of the distribution of tests for a '10-Fold' cross-validation [5].

### 3.3 Performance metrics
3.3.1 MSE It is probably the most commonly used method to predict the error. It allows evaluating the performance of multiple models on a prediction problem when dealing with continuous data.
It varies in a range of [0, inf], with a lower value of MSE, a better performance of the model. Its formulation is as follows:

MSE = (p1 - a1)^2+...+(pn-an)^2 / n
Where 'a' is the current actual value and 'p' is the predicted value [5].

3.3.2 Classification rate
The simplest way to evaluate a model with nominal and discrete characteristics is through the classification rate, whose formulation is as follows [5]:

Ranking Rate = 1 - (Wrong Rankings / Total Predictions Made)

3.3.3 Binary confusion matrix
There can only be four different types of results that give the confusion matrix, shown in fig. two:
- True positive (TP) - It is expected to have a positive value of its characteristic and a positive value is obtained as well.
- True Negative (TN) - A negative value of your characteristic is expected and a negative value is predicted as well.
- False positive (FP) - There is a negative value despite having predicted a positive value.
- False negative (FN) - There is a positive value despite having predicted a negative value [5].

![alt text](https://github.com/Juan-Ignacio-Ortega/DecisitionTree/blob/main/MatrizDConfusion.png?raw=true)

Figure 2. Confusion matrix for error calculation [5].

3.3.4 Accuracy
The classification rate can be calculated in a more precise way with the confusion matrix. It can be calculated as a model accuracy metric, whose formulation is as follows [5]:

Accuracy = (TP+TN) / (TP+TN+FP+FN)

3.3.5 Accuracy
It can be defined as the rate of the predicted samples that are relevant, it is calculated as follows [5]:

Accuracy = PT / PT+FP

3.3.6 Sensitivity (Recall)
Sensitivity, also called recall, can be defined as the rate of selected samples that are relevant to the test. It is obtained as follows:

Sensitivity = TP / TP+FN

3.3.7 F1 score
The F1 Score, F-beta score with a beta of 1, can be defined as the harmonic mean between recall and precision, and is calculated as shown in the following equation:

F1 Score = 2*TP / 2*TP+FP+FN

### 3.4 Concepts in data management
3.4.1 Quartiles
The median divides the sample in half, the quartiles divide it, as much as possible, into fourths.
Firstquartile = 0.25 * (n + 1)
Secondquartile = 0.5 * (n + 1) -> Identical to the median
Third quartile = 0.75 * (n + 1)

The result tells you the number of the value that represents the X quartile, of the data ordered in ascending order. Only if the result is an integer, if not, the average of the sample values ​​on either side of this value is taken, taking the sample in ascending order [5].

3.4.2 Data normalization
Some AI algorithms require all data to be centered around a specific range of values, typically -1 to 1 or 0 to 1. Even if the data is not required to be within the values, it is generally a good idea to ensure that the values ​​are within a specific range. Normalization of ordinal values ​​To normalize an ordinal set, the order must be preserved.

Normalization of quantitative values
The first thing you have to do is observe the range in which these values ​​are found and the interval to which you want to normalize. Not all values ​​need to be normalized.

It is necessary to perform the calculations of the following variables to find the normalized value:
1. Maximum of the data = The highest value of the observation without normalizing.
2. Minimum of the data = The lowest value of the observation without normalizing.
3. Normalized Maximum = The highest bound value to which the maximum of the data is normalized.
4. Normalized Minimum = The lowest bounding value to which the minimum of the data is normalized.
5. Range of the data = Maximum of the data - Minimum of the data
6. Normalized Range = Normalized Maximum - Normalized Minimum
7. D = Valuenormalize - Minimum of the data
8. DPct = D / Data range
9. dNorm = NormalizedRange * DPct
10. Normalized = Normalized Minimum + dNorm

In this way, the normalized value [5] is obtained.

### 3.5 confidence interval
3.5.1 Definition
A confidence interval is a range of values ​​in which we are fairly certain that our true value lies [4].

3.5.2 Calculation of the confidence interval
Step 1: Start with - The number of observations n - The mean X - The Standard Deviation s.

Step 2: Decide what confidence interval you want: 95% or 99% are common choices. Then find the 'Z' value for that confidence interval here:

Confidence Interval of:
80% - Z = 1,282
85% - Z = 1,440
90% - Z = 1.645
95% - Z = 1,960
99% - Z = 2.576
99.5% - Z = 2,807
99.9% - Z = 3,291

Step 3: Use that Z value in this formula for the confidence interval
X ± (Z * s/root(n))
Where:
- X is the mean
- Z is the Z value chosen from the table above
- s is the standard deviation
- n is the number of observations

The value after the ± is called the margin of error [4].

## 4 - References
[1] 1.10. Decision Trees. (s. f.). scikit-learn. Recuperado 10 de junio de 2022, de https://scikitlearn/
stable/modules/tree.html

[2] Bhardwaj, R., & Vatta, S. (2013). Implementation of ID3 algorithm. International Journal of
Advanced Research in Computer Science and Software Engineering, 3(6).

[3] Gómez, P. M. (2021). Drugs A, B, C, X, Y for decision trees [Data set].

[4] Intervalo de Conanza. (s. f.). Disfrutalasmatematicas.com. Recuperado 10 de junio de 2022,
de http://www.disfrutalasmatematicas.com/datos/intervalo-conanza.html

[5] M. A. Aceves Fernández, Inteligencia Articial para programadores con prisa. UNIVERSO de
LETRAS, 2021.

[6] Quinlan, J. R. (1996). Learning decision tree classiers. ACM Computing Surveys (CSUR),
28(1), 71-72.
