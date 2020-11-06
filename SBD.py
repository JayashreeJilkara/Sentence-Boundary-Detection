import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def data_preprocessing(path):
    data = []
    with open(path, 'r') as file:
        for line in file.readlines():
            if len(line) > 2:
                data.append(line.split())
    all_features = []
    for i in range(0, len(data) - 1):
        feature = []
        current_element = data[i]
        next_element = data[i + 1]
        if current_element[2] == 'EOS' or current_element[2] == 'NEOS':
            left_word = current_element[1].split('.')[0]
            right_word = next_element[1].split('.')[0]
            feature.append(left_word)
            feature.append(right_word)
            feature.append(1 if len(left_word) < 3 else 0)
            feature.append(1 if left_word.isupper() else 0)
            feature.append(1 if right_word.isupper() else 0)
            feature.append(1 if right_word.isalpha() else 0)
            feature.append(1 if left_word.isalpha() else 0)
            feature.append(1 if right_word.isdigit() else 0)
            feature.append(1 if current_element[2] == 'EOS' else 0)
            all_features.append(feature)
    return all_features

train = sys.argv[1]
test = sys.argv[2]
train_processed = data_preprocessing(train)
test_processed = data_preprocessing(test)
train_df = pd.DataFrame(train_processed)
test_df = pd.DataFrame(test_processed)
full = pd.concat([train_df, test_df])
le = LabelEncoder()
le.fit(full[0])
train_df[0] = le.transform(train_df[0])
test_df[0] = le.transform(test_df[0])
le2 = LabelEncoder()
le2.fit(full[1])
train_df[1] = le2.transform(train_df[1])
test_df[1] = le2.transform(test_df[1])
labels = train_df[8]
train_df = train_df.drop(columns=[8])
train_df_2 = train_df.drop(columns=[5, 6, 7])
test_df_2 = test_df.drop(columns=[5, 6, 7, 8])
train_df_3 = train_df.drop(columns=[3, 4, 5])
test_df_3 = test_df.drop(columns=[3, 4, 5, 8])
dtc = DecisionTreeClassifier()
dtc.fit(train_df, labels)
test_labels = test_df[8]
test_df = test_df.drop(columns=[8])
pred = dtc.predict(test_df)
print("Accuracy = " + str(dtc.score(test_df, test_labels)*100) + " %")
test_df[0] = le.inverse_transform(test_df[0])
test_df[1] = le2.inverse_transform(test_df[1])
test_df[8] = pred
test_df[8] = test_df[8].apply(lambda x: 'EOS' if x == 1 else 'NEOS')
test_df = test_df.drop(columns=[2, 3, 4, 5, 6, 7])
test_df.columns = ['Left word', 'Right word', 'Prediction']
test_df.to_csv('SBD.test.out')
dtc2 = DecisionTreeClassifier()
dtc2.fit(train_df_2, labels)
print("Accuracy of model with only given features = " + str(dtc2.score(test_df_2, test_labels)*100) + " %")
dtc3 = DecisionTreeClassifier()
dtc3.fit(train_df_3, labels)
print("Accuracy of model with only my features = " + str(dtc3.score(test_df_3, test_labels)*100) + " %")
