import tensorflow as tf
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()

feature_columns = skflow.infer_real_valued_columns_from_input(iris.data)

classifier = skflow.LinearClassifier(feature_columns=feature_columns, n_classes=3)

classifier.fit(x=iris.data, y=iris.target, steps=20000)

predictions = list(classifier.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, predictions)

print("Accuracy: %f" %score)