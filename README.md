# SYLLABUS

```text
          ┌──────────────────────────────────────────────┐
          │            Artificial Intelligence (AI)      │
          │                                              │
          │  ┌────────────────────────────────────────┐  │
          │  │          Machine Learning (ML)         │  │
          │  │                                        │  │
          │  │  ┌──────────────────────────────────┐  │  │
          │  │  │      Supervised Learning         │  │  │
          │  │  │  (Regression, Classification)    │  │  │
          │  │  └──────────────────────────────────┘  │  │
          │  │                                        │  │
          │  │  ┌──────────────────────────────────┐  │  │
          │  │  │     Unsupervised Learning        │  │  │
          │  │  │  (Clustering, PCA, etc.)         │  │  │
          │  │  └──────────────────────────────────┘  │  │
          │  │                                        │  │
          │  │  ┌──────────────────────────────────┐  │  │
          │  │  │   Reinforcement Learning (RL)    │  │  │
          │  │  │                                  │  │  │
          │  │  │   ┌──────────────────────────┐   │  │  │
          │  │  │   │ Deep Reinforcement       │   │  │  │
          │  │  │   │ Learning                 │   │  │  │
          │  │  │   └──────────────────────────┘   │  │  │
          │  │  └──────────────────────────────────┘  │  │
          │  │                                        │  │
          │  │  ┌──────────────────────────────────┐  │  │
          │  │  │        Deep Learning (DL)        │  │  │
          │  │  │                                  │  │  │
          │  │  │  ┌────────┐  ┌───────────────┐   │  │  │
          │  │  │  │  MLP   │  │     CNN       │   │  │  │
          │  │  │  └────────┘  └───────────────┘   │  │  │
          │  │  │  ┌────────┐  ┌───────────────┐   │  │  │
          │  │  │  │  RNN   │  │ Transformers  │   │  │  │
          │  │  │  └────────┘  └───────────────┘   │  │  │
          │  │  │  ┌──────────────────────────┐    │  │  │
          │  │  │  │ Generative Models        │    │  │  │
          │  │  │  └──────────────────────────┘    │  │  │
          │  │  └──────────────────────────────────┘  │  │
          │  │                                        │  │
          │  └────────────────────────────────────────┘  │
          │                                              │
          └──────────────────────────────────────────────┘
```

This repository documents my learning journey in **Machine Learning**, covering fundamental concepts, hands-on implementations, and practical experiments.  
It includes topics from **supervised learning**, **unsupervised learning**, **reinforcement learning**, and **deep learning with neural networks**.

<div class="rl-section">
  <h2>Reinforcement Learning</h2>

  <p>
    <strong>Reinforcement Learning</strong> is a type of machine learning where an agent learns
    by interacting with an environment and improving its behavior through
    <strong>rewards</strong> and <strong>penalties</strong>.
  </p>

  <p>
    For example, a Pac-Man agent can be designed using
    <strong>State (S)</strong>, <strong>Action (A)</strong>,
    <strong>Reward (R)</strong>, and <strong>Policy (π)</strong>
    to guide its decisions and learn the most efficient way to move and win the game.
  </p>

  <h4>Common Reinforcement Learning Algorithms</h4>
  <ul>
    <li><strong>Q-Learning</strong> — off-policy value-based learning</li>
    <li><strong>SARSA</strong> — on-policy value-based learning</li>
  </ul>
</div>

---

<div class="dl-section">
  <h2>Deep Learning</h2>
  <h5>Gradient Descent</h5>
  <p>Gradient Descent is an optimization algorithm used to train machine learning models by minimizing a loss (error) function.Gradient descent repeatedly adjusts model parameters in the direction that most reduces error.</p>

  <h5>Autodiff</h5>
  <p>Autodiff computes derivatives of functions by systematically applying the chain rule to the operations that produced them. It is basically a calculus trick for finding the gradients in gradient dscent.</p>

  <h5>Softmax</h5>
  <p>Softmax is uesd for classification. Given a score for each class, it produces a probability of each class. The class with the highest probability is the "answer" you get. </p>

  <h5>LTU(1957)</h5>
  <p>An LTU takes several inputs, combines them linearly multiplied by weights and adds Bias Neuron, and then applies a threshold to decide the output.</p>

  <h5>Perceptron</h5>
  <p>A perceptron is the simplest form of a neural network unit, and at its core it is a Linear Threshold Unit (LTU). Used Step function.</p>

  <h5>Multi-layer Perceptron</h5>
  <p>Addition of "hidden layers" which is known as Deep Neural Network.</p>

  <h5>A Modern Deep Nueral Network</h5>
  <ul>
  <li>Uses differentiable activation functions (e.g., ReLU) instead of step functions</li>
  <li>Learns weights through backpropagation + gradient descent</li>
  <li>Produces probabilistic outputs using softmax in the final layer</li>
  <li>Can model non-linear and complex relationships by stacking many layers</li>
  </ul>

  <h5>Backpropagation</h5>
  <p>Backpropagation is gradient descent implemented via reverse-mode automatic differentiation. The algorithm first computes the output error, then determines how much each neuron in the previous layers contributed to that error by propagating gradients backward through the network. These gradients are then used to update the weights via gradient descent, reducing the error. </p>

  <h5>Activation Functions</h5>
  <p>An activation function transforms a neuron’s weighted input into an output that introduces nonlinearity, allowing neural networks to learn complex patterns instead of just straight-line relationships. Without activation functions, deep networks would collapse into a single linear model. Different activations control how neurons fire and how well the network can learn, with modern models relying on smooth, gradient-friendly functions like Rectified Linear Unit(ReLU). Alternatives are Logistic function, Hyperbolic tangent function, Exponential linear unit(ELU).</p>

<h5>Optimization Functions</h5>
<ul>
  <li><strong>Momentum</strong> – Accelerates gradient descent by accumulating past gradients, helping the model move faster in consistent directions.</li>
  <li><strong>Nesterov Accelerated Gradient (NAG)</strong> – Improves momentum by looking ahead at the future position before computing the gradient.</li>
  <li><strong>RMSProp</strong> – Adapts the learning rate for each parameter by normalizing gradients using a moving average of squared gradients.</li>
  <li><strong>Adam</strong> – Combines momentum and RMSProp to provide fast, stable, and adaptive learning.</li>
</ul>

<h5>Avioid Ovefritting</h5>
<p>This can be done by Early Stopping, Adding Regularization terms to cost function, Dropout(igonores 50% of the neurons randonmly at each tarining step.)</p>

  <h6>Useful Website: playground.tensorflow.org</h6>

</div>

---

<div>
  <h3>Tensorflow</h3>
  <p>Tensorflow is made by Google and it is generally an architecture for executing a graph of numerical operations. Tensorflow can optimize the processing of that graph, and distribute its processing across a network. Tensorflow can distribute work across GPU Cores. In addition, it can be run on about everything(phone, computer) and it is highly efficient C++ code with easy to use Python API's.</p>
  <pre>
  <code>import tensorflow as tf
  a = tf.Variable(1, name="a")
  b = tf.Variable(2, name="b")
  f = a+b
  tf.print(f)</code></pre>

<h5>Creating a neural network with tensorflow</h5>
<ul><li>Load up our training and testing data</li>
<li>Construct a graph describing our neural network</li>
<li>Associate an optimizer(ie gradient descent) to the network</li>
<li>Normalize the input data(training, test)</li>
<li>Run the optimizer with your training data</li>
<li>Evaluate your trained network with your testing data</li>
</ul>

<h5>Keras</h5>
<p>Keras is a high-level deep-learning API that lets you build, train, and evaluate neural networks easily and safely, without writing low-level math or training loops. It also has scikit_learn integration.</p>
<pre><code>from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

model = sequencial()
model.add(Input(shape=(20,)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Desne(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy', optmizer=sgd, metrics=['accuracy'])

estimator = KerasClassifier(model=create_model, epochs=100, verbose=0)
cv_scores = cross_val_score(estimator, features, labels, cv=10)

Use loss='binary_crossentropy', activatyion='sigmoid' for binary classification

</code></pre>

</div>

---

<div>
<h5>CNN(Convolutional Nueral Networks)</h5>
<p>A Convolutional Neural Network (CNN) is a deep learning model designed for grid-like data such as images. Instead of connecting every pixel to every neuron, CNNs focus on small local regions, allowing them to learn patterns like edges, textures, and shapes. They are powerful because they understand where features appear in an image, use far fewer weights through parameter sharing, and can recognize objects even when they move slightly within the image.</p>
<ol>
  <li>Input – An image is given as a grid of pixel values.</li>
  <li>Convolution – Small filters slide over the image to detect local patterns such as edges and textures, producing feature maps.</li>
  <li>Activation (ReLU) – Keeps useful signals and adds non-linearity so the model can learn complex patterns.</li>
  <li>Pooling – Reduces the size of feature maps by keeping the most important information, improving efficiency and robustness.</li>
  <li>Hidden Layers (Feature Hierarchy) – As convolution, ReLU, and pooling repeat, early hidden layers learn edges and lines, while deeper hidden layers combine them to define corners, textures, and meaningful shapes.</li>
  <li>Flatten – Converts the final feature maps into a single vector.</li>
  <li>Fully Connected Layers – Combine the learned shapes and features to make a decision.</li>
  <li>Output – Produces the final prediction (e.g., class probabilities).</li>
  </ol>

  <pre><code>With Keras
  Conv2D->MaxPooling2D->Dropout->Flatten->Dense->Dropout->Softmax</code></pre>

  <p>When applying a CNN layer, we specify the number of kernels (for example, 32) and the kernel size (such as 3×3). Each kernel slides over the input and performs convolution, producing one feature map. As a result, using 32 kernels generates 32 feature maps, each slightly smaller in height and width when no padding is used. These feature maps capture different local patterns such as edges and textures.

In the next CNN layer, we may use a larger number of kernels (for example, 64) with the same kernel size. Each of these kernels now operates on all 32 feature maps produced by the previous layer. For every spatial position, a kernel combines information from the corresponding regions of all 32 feature maps, sums the results, and produces a single output value. Repeating this process across the entire input creates one output feature map per kernel, resulting in 64 new feature maps that represent more complex patterns built from earlier features.</p>

</div>

---

<div>
<h5>RNN(Recurrent Nueral Networks)</h5>
<p>RNN (Recurrent Neural Network) is a type of neural network designed to work with sequential data, where the order of data points matters. Unlike standard neural networks that treat each input independently, an RNN keeps a hidden state (memory) that carries information from previous time steps, allowing the model to learn patterns over time.

RNNs are used when past information is important for understanding the present. Common applications include natural language processing (such as text prediction, translation, and sentiment analysis), speech recognition, time-series forecasting (like stock prices or sensor data), and sequence modeling tasks where context and order are crucial. In short, RNNs are for problems that involve data unfolding over time and require the model to remember what came before.

Long Short-Term Memory Cell(MSTM) Cell and Gated Recurrent Unit(GRU) can be used to train RNNs.

</p>

<ul>
<li>Sequence to sequence</li>
<li>Sequence to vector</li>
<li>vector to sequence</li>
<li>Encoder -> Decoder ( Sequence->vector->sequence)</li>
</ul>

<h6>Backpropagation in an RNN is called Backpropagation Through Time (BPTT) because the network is unrolled across time steps, and gradients are sent backward through those time steps.<h6>
</div>
---

### Measuring Classifiers

#### Basic Confusion Matrix (Binary Classification)

|                                     | <strong>Actual Positive</strong> | <strong>Actual Negative</strong> |
| ----------------------------------- | -------------------------------- | -------------------------------- |
| <strong>Predicted Positive</strong> | TP (True Positive)               | FP (False Positive)              |
| <strong>Predicted Negative</strong> | FN (False Negative)              | TN (True Negative)               |

#### Recall (Sensitivity / True Positive Rate)

<strong>Recall</strong> = TP / (TP + FN)

#### Precision (Correct Positive Rate)

<strong>Precision</strong> = TP / (TP + FP)

#### Specificity (True Negative Rate)

<strong>Specificity</strong> = TN / (TN + FP)

#### F1 Score

<strong>F1 Score</strong> = 2TP / (2TP + FP + FN)  
or  
<strong>F1 Score</strong> = 2 × (Precision × Recall) / (Precision + Recall)

#### Other Evaluation Metrics

- <strong>ROC Curve</strong>
- <strong>AUC Score</strong>

---

<div>
  <h3>Bias and Variance</h3>
  <p>
    <strong>Bias</strong> measures how far the average predicted values are from the
    <strong>true value</strong>.
  </p>
  <p>
    <strong>Variance</strong> measures how much the predictions vary around their
    average.
  </p>
  <p>
    <strong>Error = Bias² + Variance</strong>
  </p>
</div>

---

<div>
  <h3>K-Fold Cross Validation</h3>
  <p>
    <strong>K-Fold Cross Validation</strong> splits the dataset into <strong>K</strong> parts,
    trains the model on <strong>K−1</strong> parts, tests it on the remaining part,
    and repeats this process until every part has been used for testing.
  </p>
  <pre><code class="language-python">
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
  </code></pre>
</div>

---

<div>
  <h3>Cleaning Your Data</h3>
  <p>
    In real-world scenarios, data often contains
    <strong>outliers</strong>, <strong>missing data</strong>,
    <strong>malicious data</strong>, <strong>erroneous data</strong>,
    <strong>irrelevant data</strong>, <strong>inconsistent data</strong>,
    and <strong>formatting issues</strong>.
  </p>
</div>

---

<div>
  <h3>Normalize Your Data</h3>
  <p>
    Input data often needs to be <strong>normalized</strong>, <strong>scaled</strong>,
    or <strong>whitened</strong>. For example, categorical values such as
    "yes" and "no" must be converted to numeric values like "1" and "0".
    A best practice is to always consult the documentation to determine
    whether normalization is required.
  </p>
</div>

---

<div>
  <h3>Dealing with Outliers</h3>
  <p>
    In a movie recommender system, <strong>power users</strong> who have watched and rated
    nearly every movie can act as <strong>outliers</strong>. These outliers should be
    removed cautiously, depending on the desired outcome of the model.
  </p>
  <p>
    Common techniques include statistical approaches such as
    <strong>standard deviation</strong>.
  </p>
</div>

---

<div>
  <h3>Feature Engineering</h3>
  <p>
    <strong>Feature engineering</strong> is the process of designing, transforming,
    and selecting input variables so that a machine learning model can learn
    patterns more effectively. This is especially important due to
    <strong>the curse of dimensionality</strong>, where too many features can
    degrade model performance.
  </p>
  <p>
    Unsupervised dimensionality reduction techniques such as
    <strong>PCA</strong> and <strong>K-Means</strong> can be employed to distill
    many features into fewer, more informative features.
  </p>

  <h5>Binning</h5>
    <p><strong>Binning</strong> is a feature engineering technique that converts a continuous numerical feature into discrete intervals (bins).</p>
  <h5>Transforming</h5>
    <p><strong>Transforming</strong> means changing the graph representation(eg. exponential, linear) of features so that a machine learning model can learn patterns more effectively.</p>
  <h5>Encoding</h5>
    <p><strong>Encoding</strong> is the process of converting categorical data into numerical values so that machine learning models can understand and use them. For Example, One-hot encoding creates "buckets" for every category in which the bucket for my category is 1 and all other are 0.</p>
  <h5>Scaling/Normalization</h5>
    <p><strong>Scaling (Normalization)</strong> is the process of rescaling numerical features so they share a similar range or distribution, preventing features with large values from dominating the model.</p>
  <h5>Shuffling</h5>

</div>

---

<div>
  <h3>Imputing Missing Data</h3>

  <h5>Mean Imputation</h5>
  <p>
    <strong>Mean imputation</strong> replaces missing values in a numerical feature
    with the <strong>mean</strong> of the observed values. While simple and efficient,
    it may reduce variance and distort the data distribution.
    The <strong>median</strong> can be used when outliers are present.
  </p>

  <h5>Dropping</h5>
  <p>
    <strong>Dropping</strong> missing data removes observations or features that
    contain missing values. Although simple, it can reduce dataset size and
    introduce bias if the missingness is not random.
  </p>

  <h5>KNN Imputation</h5>
  <p>
    <strong>KNN imputation</strong> fills missing values using the most similar
    data points based on feature distance. It preserves local structure but is
    computationally expensive and sensitive to feature scaling.
  </p>

  <h5>Regression Imputation</h5>
  <p>
    <strong>Regression imputation</strong> predicts missing values using a regression
    model trained on observed data. An advanced method is
    <strong>MICE (Multiple Imputation by Chained Equations)</strong>.
  </p>

  <h5>Deep Learning</h5>
  <p>
    <strong>Deep learning-based imputation</strong> uses neural networks to model
    complex feature relationships. These methods can be highly accurate but require
    large datasets and significant computational resources.
  </p>

  <p><strong>Getting more data is often the best solution.</strong></p>
</div>

---

<div>
  <h3>Unbalanced Data</h3>
  <p>
    <strong>Unbalanced data</strong> occurs when one class is significantly more
    frequent than others, which is common in domains such as fraud detection and
    medical diagnosis. This can bias models toward the majority class.
    Common solutions include <strong>oversampling</strong>,
    <strong>undersampling</strong>, <strong>SMOTE</strong>, and
    <strong>threshold adjustment</strong>.
  </p>
</div>

---

<div>
<h3>SPARK</h3>
<p>
  <strong>Apache Spark</strong> is a fast, distributed data processing engine designed to efficiently handle very large datasets.
  A Spark application starts with a <em>Driver Program</em>, which runs user code and creates a
  <code>SparkContext</code> (or <code>SparkSession</code>). The driver analyzes the program, builds an execution plan,
  and divides the work into smaller units called tasks. The <code>SparkContext</code> then communicates with a
  <em>Cluster Manager</em> (such as Standalone Spark or YARN) to request computing resources like CPU and memory.
  The cluster manager allocates these resources and launches <em>executors</em> on worker nodes. These executors
  run tasks in parallel, each processing a partition of the data, and can cache data in memory to improve
  performance, allowing Spark to scale from a single machine to large clusters while maintaining high speed.
  Spark consists of several major components, including <strong>Spark Core</strong>, <strong>Spark SQL</strong>,
  <strong>Spark Streaming</strong> (Structured Streaming), <strong>MLlib</strong>, and <strong>GraphX</strong>.
</p>

  <h5>Resilient Distributed Datasets (RDD)</h5>
  <p>An <strong>RDD</strong> is Spark’s core data structure: an immutable, distributed collection of data that is split across many machines and processed in parallel.</p>
  <pre><code class="language-python">
    nums = parallelize([1,2,3,4])
    sc.text("file:///c:/Users/peihe/gobs-o-text.txt")
    hiveCtx = HiveContext(sc) rows = hiveCtx.sql("SELECT name, age FROM users")
    Can also create from: JDBC, Cassandra, HBase,
    Elastisearch,JSON, CSV, sequence files, 
    object files, various compressed formats
  </code></pre>

  <p><strong>Transforming RDD's</strong>: map, flatmap, filter, distinct, sample, union, intersection, subtract, cartesian</p>
  <pre><code class="language-python">
    rdd = sc.parallelize([1,2,3,4])
    rdd.map(lambda x: x*x)
    rdd.(collect, count, countByValue, take, top, reduce, ..and more ...)
  </code></pre>

  <h5>MLLib</h5>
  <p>MLlib is Apache Spark’s built-in machine learning library. Supports Feature Extraction, Basic statistics, Linear Regression, Vector Machines, naive Bayes Classifier, Decision Trees, K-Means Clustering, and etc. </p>
  <p>It also introduces special MLLib Data Types: Vector, LabeledPoint, Rating.</p>
  <pre>
  <p>To run Spark in Terminal:</p>
    <code>Spark-submit file.py</code> 
  </pre>

  <h5>TF-IDF</h5>
  <p>Stands for Term Frequency and Inverse Document Frequency, which figures our what terms are most relevant for a document.</p>
  <code>Relevance: Term Frequency / Document Frequency</code>
</div>

---

<div>
<h3>Deploying Models to Real-Time Systems</h3>
<h5>Google Cloud ML</h5>
<p>Durmp your trained classifier using sklearn.externals and upload model.joblib to google cloud storage, specifying the scikit-learn framework. Then Cloud ML Engine exposes a REST API to call model to make predictions in real time.</p>
<pre>
<code>from sklearn.externals import joblib
joblib.dump(clf, 'model.joblib')
</code>
</pre>
</div>

---

<div>
<h3>A/B test</h3>
<p>The process of testing the performance of some change to yor website(the variant) and measure conversion relative to your unchanged site(the control) controlled experiment, usually in the context of a website. Variance is enemy!</p>
<ul>
<li>Design changes</li>
<li>UI flow</li>
<li>Algorithmic changes</li>
<li>Pricing changes</li>
<li>etc</li>
</ul>

<h5>T-Test and P-Values</h5>
<p>T-Test is the measure of the difference between the two sets expressed in units of standard error. A high t-value means there's probably a real differnce between the two sets.</p>
<p>P-Value is the probability of A and B satisfying the "null hypothesis." A low P-Value implies significance. For example, if I was given a coin and say "This is a fair coin." which is null hypothesis. Now ask "If this coin is truly fair, how weird is seeing 10 heads?" "How weird" is P-Value here.
</p>
<ul>
<li>Fisher's exact test(for clickthrough rates)</li>
<li>E-test(for transactions per user)</li>
<li>Chi-sqaured test(for product quantities purchased)</li>
</ul>
<pre><code>from scipy import stats
stats.ttest_ind(A,B)</code></pre>
</div>

---

<div>
  <h3>Libraries</h3>
  <ul>
    <li><strong>pandas</strong></li>
    <li><strong>seaborn</strong></li>
    <li><strong>matplotlib</strong></li>
    <li><strong>scikit-learn</strong></li>
  </ul>
</div>

<div>
    <h3>Tools</h3>
        <ul>
        <li>Anaconda</li>
        <li>Spark</li>
        <li>JDK</li>
    </ul>
</div>

---

## Learning Resource

This repository is inspired by and based on the Udemy course:

<strong>Machine Learning, Data Science & AI Engineering with Python</strong>  
🔗 https://www.udemy.com/share/101Y8w3@knal-tXJn4bqDdIVmf7J149ZK1uEXz_ap_FDVXvkuAqSPjc1bW_rYDNFnsN7lvp01g==/
