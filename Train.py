import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

train_file_path = 'train.csv'
dataset = tf.data.experimental.make_csv_dataset(train_file_path, batch_size=32,
                                                label_name='Survived', na_value="?",
                                                num_epochs=1, ignore_errors=True)


feature_columns = []
for key in dataset.keys():
    if key != 'Survived':
        feature_columns.append(tf.feature_column.numeric_column(key))


def preprocess(features, labels):
    labels = labels['Survived']
    return features, labels

dataset = dataset.map(preprocess).shuffle(buffer_size=1000)


train_dataset = dataset.take(700)
test_dataset = dataset.skip(700)


clf = RandomForestClassifier(n_estimators=100, random_state=42)

for features, labels in train_dataset:
    features_dict = {}
    for i, key in enumerate(dataset.keys()):
        if key != 'Survived':
            features_dict[key] = features[:, i]
    clf.partial_fit(features_dict, labels, classes=[0, 1])

train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(f"Training accuracy: {train_score:.3f}, Testing accuracy: {test_score:.3f}")

joblib.dump(clf, 'titanic_model.pkl')
