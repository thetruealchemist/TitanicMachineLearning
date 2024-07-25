"Titanic - Machine Learning from Disaster" - is a competition on Kaggle where we have to predict the survival chances of some of the passengers of Titanic based on the data of other passengers that survived and did not survive.

-> Competition Link: https://www.kaggle.com/competitions/titanic

-> We have been provided 2 files, namely train_row.csv, and test_row.csv, and we have to train our model on the data of train_row.csv and predict the survival of test_row.csv.

-> All the Excel files that I created during the work process are to be found in the folder called "excel_files", similarly all python(Jupyter) files in "python_files" and the "state_dict()" of one of the models that I trained is in "models" folder.

-> So this is how I created this model in chronological order:

-> Note: Please open the respective .ipynb files for reference.

-> Note: Python Version Used: 3.11.9 and Pytorch Version Used: 2.4.0

Dependencies:

```python
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
```

1. File Name: data_processing.ipynb

```python
df1 = pd.read_csv('../excel_files/train_row.csv')
df2 = pd.read_csv('../excel_files/test_row.csv')
df = pd.concat([df1, df2], ignore_index=True)
df = df.drop(columns=['Survived', 'Cabin', 'Name', 'Ticket'])
df['Embarked'] = df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2, np.nan: 0})
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
df['Alone'] = ((df['Parch'] == 0) & (df['SibSp'] == 0)).astype(int)
df.to_csv('../excel_files/dataset_01.csv', index=False)
```

-> Imported both of the files, merged them vertically, and dropped the columns 'Cabin', 'Name', and 'Ticket' that I thought did not play an important role in prediction and Cabin because almost all of the values in train_dataset were missing, so it is not useful anyways and finally 'PassengerId' because it also does not play any role in training.

-> Now I had to convert alphanumerical data into numerical data so that it is easier for me to train the model, firstly I converted the 'Embarked' column elements 'S', 'C', and 'Q' into 0, 1, and 2 respectively. Also, there were some missing values which I also filled with 0 as in 'S' only because we had the clear majority (mod) of S in the train and test dataset.

-> Then I filled the missing values from 'Fare' column with the mean of the rest of the available values. In the 'Sex' column, I converted 'male' to 0 and 'female' to 1.

-> Then I did something interesting, that is, I added an artificial column named 'Alone' and I gave it the value of 1 in case the value of 'Parch' AND 'SibSp' is 0 simultaneously, otherwise if the value in either of the two columns is non-zero then the value in 'Alone' column should be 0. The reason I did this is because as per my understanding and research, the people who were completely alone on the sip without any family member had significantly higher chances of survival as compared to their contrary. The value of 1 in Alone therefore shows that the person was travelling alone.

-> Saved the .csv file as 'dataset_01.csv'.

2. File Name: scaling_values.ipynb

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
df = pd.read_csv("../excel_files/dataset_01.csv")
scaler = MinMaxScaler()
df['Fare'] = scaler.fit_transform(df[['Fare']])
age_missing = df[df['Age'].isnull()]
age_not_missing = df.dropna(subset=['Age'])
X = age_not_missing[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Alone']]
y = age_not_missing['Age']
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)
```
-> Device agnostic code for GPU, in my case since I was using Mac, I had "mps". If you are using any other type of GPU, you will have to change it accordingly, otherwise thanks to this device agnostic code my code will still work but on CPU which would be extremely slow FYI.

-> MinMaxScaler from sklearn to scale the values of 'Fare' to help in machine learning later. Then I thought of a way to predict the missing values of 'Age' because in my opinion, it was the second most important factor in prediction after gender, so I had to do it very carefully. 

-> So I first built a Linear Regression Model based on the six factors, namely 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', and 'Alone', which I thought would play a role in deciding the age of the person. In this part of the code I first separeted the missing and non-missing values, then turned the non-missing values into torch.tensor type and then split the tensor into training and testing set with 80% and 20% of the original tensor respectively.

```python
class AgePredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=X_tensor.shape[1], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=len(y_tensor.shape))
        )

    def forward(self, x):
        return self.layer_stack(x)
```
-> Here I built a model with Linear Layers along with ReLU layers in the middle. The reason I decided on this particular combination is that I did a lot of testing with multiple combinations and this combination gave the best results in the smallest amount of time.

```python
torch.manual_seed(42)
age_prediction_model = AgePredictionModel()
age_prediction_model.to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(age_prediction_model.parameters(), lr=0.01)
torch.manual_seed(42)
epochs = 10000
for epoch in range(epochs):
    age_prediction_model.train()
    y_pred = age_prediction_model(X_train).squeeze()
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    age_prediction_model.eval()
    with torch.inference_mode():
        test_pred = age_prediction_model(X_test).squeeze()
        test_loss = loss_fn(test_pred, y_test)

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}] | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}')
```
-> Here I used L1Loss aka MAE Loss, I also tried MSE Loss at first but L1 Loss gave exponentially better results. For the optimizer, I used typical Adam. I then trained the model on the train_dataset and then evaluated it on the test_dataset and printed out the results.

-> With 10000 Epochs, I was able to get 7.4 Train Loss and 8.8 Test Loss which seemed reasonable and good to go.

```python
X_missing = age_missing[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Alone']]
X_missing_tensor = torch.tensor(X_missing.values, dtype=torch.float32).to(device)
age_prediction_model.eval()
with torch.inference_mode():
    predicted_age_tensor = age_prediction_model(X_missing_tensor)
    predicted_age = predicted_age_tensor.cpu().numpy().flatten()
predicted_age.shape
df.loc[df['Age'].isnull(), 'Age'] = predicted_age
df.to_csv('../excel_files/dataset_02.csv', index=False)
df = pd.read_csv('../excel_files/dataset_02.csv')
df['Age'] = scaler.fit_transform(df[['Age']])
df.to_csv('../excel_files/dataset_03.csv', index=False)
```
-> I then predicted the missing age values, put them in the dataset, and saved the .csv file as 'dataset_02.csv'. Then I loaded the data frame again, used the same MinMaxScale from Sklearn on 'Age' values as well, and then saved the file as 'dataset_03.csv'.

3. File Name: data_processing_02.ipynb

```python
df = pd.read_csv('../excel_files/dataset_03.csv')
df1_old = pd.read_csv('../excel_files/train_row.csv')
df1 = df.iloc[:891]
df2 = df.iloc[891:]
df1['Survived'] = df1_old['Survived'].values
df1.to_csv('../excel_files/dataset.csv', index=False)
df2.to_csv('../excel_files/result.csv', index=False)
```

-> In this file, I just separated the data merged in 'data_processing.ipynb' again into 2 parts, 'dataset.csv' which is going to be used for training the model and 'result.csv' where I have to make the prediction. 891 as separating number is because originally I had 891 rows in the first file before combining. But now that I think of it, I should not have hard-coated the value number of rows, to make the code more robust, but still it is the correct value and works absolutely fine.

4. File Name: main.ipynb

```python
dataset = pd.read_csv("../excel_files/dataset.csv")
X = dataset.drop(columns=['Survived', 'PassengerId'])
y = dataset['Survived']
X_tensor = torch.tensor(X.values, dtype=torch.float).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float).to(device)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
class TitanicModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=X_train.shape[1], out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=len(y_train.shape))
        )
    def forward(self, x):
        return self.layer_stack(x)
torch.manual_seed(42)
model_0 = TitanicModelV0()
model_0.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(params=model_0.parameters(), lr=0.001)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
torch.manual_seed(42)
epochs = 9000
for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # Prediction thresholding for accuracy calculation
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: [{epoch + 1}/{epochs}] | {loss = :.4f} | {acc = :.2f}% | {test_loss = :.4f} | {test_acc = :.2f}%")
MODEL_NAME = "titanic_model_0.pth"
MODEL_PATH = Path("../models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)
```
-> Here I just made a simple Linear Regression Model, nothing fancy and I REALLY mean it, trust me I tried to make the most complex models in the hope that it might work better but all of them gave similar results if not worse, most of the time I had to face overfitting of the model, where the model has given very high accuracy almost close to 99% but at the same time poor accuracy on the testing dataset.

-> So in the end, I decided to go with the rather sample model, which would save a lot of time and has very little chance of overfitting with average results. With this model, I got 90% Train Accuracy and 85% Test Accuracy after 9000 Epochs. I was satisfied with the results.

-> I chose BCEWithLogitsLoss as my loss_fn because in my experience this loss_fn performs almost always the best in binary classification problems like this one. For Optimizer I used AdamW, which performed the best as compared to the typical Adam and SGD Optimizers.

-> Sigmoid function to convert the row logits into prediction probabilities and round function to get the labels of either 1 or 0. I then saved the state_dict() of the model in the "models" folder with the name 'titanic_model_0.pth'.

```python
result_dataset = pd.read_csv("../excel_files/result.csv")
X_result = result_dataset.drop(columns=['PassengerId'])
X_result_tensor = torch.tensor(X_result.values, dtype=torch.float).to(device)
model_0.eval()
with torch.inference_mode():
    y_result_logits = model_0(X_result_tensor)
y_result_labels = torch.round(torch.sigmoid(y_result_logits))
y_result_labels = y_result_labels.cpu().numpy().astype(int)
result_dataset['Survived'] = y_result_labels
result_dataset.to_csv("../excel_files/final_results.csv", index=False)
```
-> Here I just imported the 'result.csv' data frame, predicted the 'Survived' values using my trained model 'model_0', and then saved the results as 'final_results.csv'.

5. File Name: data_processing_3.ipynb
```python
df = pd.read_csv('../excel_files/final_results.csv')
df = df.drop(columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Alone'])
df.to_csv('../excel_files/upload.csv', index=False)
```

-> Here I just imported the file 'final_results.csv', deleted the non-required columns as per the upload instructions and then saved the file of the two remaining columns named 'PassengerId' and 'Survived' as 'upload.csv'.

-> I then uploaded the file into the Kaggle competition and got a score of 0.76076, which I understand is not an ideal score, but I am pretty satisfied considering it was my first ever machine learning project other than practice problems that I did before.

-> I understand there are a lot of improvements that could be done in this model, a few of which I have already noticed while writing about it, but I thought to make these improvements on my next projects directly.

-> I would like to thank you for going through my model, I hope I was able to clearly explain my model and you were able to learn something new, as did I.

Have a nice day :) <br>
The True Alchemist.
