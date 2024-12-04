import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#model library
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
"----------------------------------------------"
# import data
data=pd.read_csv(r"C:\Users\Elbostan\Desktop\New folder\breat cancer\data.csv")

"----------------------------------------------"
# Data preprocessing
data.info()
data.describe()
data.isna().sum()
data.drop(columns=['Unnamed: 32'], inplace=True)
data.drop(columns=["id"],inplace=True)
data.shape
"-------------------------------------------------"
# Build the DNN mode

X=data.drop(columns=["diagnosis"])
y=data["diagnosis"]
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model=Sequential([
    Dense(128,activation="relu",input_shape=(30,)),
    Dense(64,activation="relu"),
    Dropout(0.5),
    Dense(32,activation="relu"),
    Dropout(0.2),
    Dense(32,activation="relu"),
    Dropout(0.2),
    Dense(1,activation='sigmoid')
])

"--------------------------------------------------"
# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
    loss="binary_crossentropy",
    metrics=['accuracy']
)


early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,  # Stop after 5 epochs without improvement
    restore_best_weights=True
)
"---------------------------------------------------"

# Train the model
history=model.fit(x_train,y_train,epochs=30,batch_size=3,validation_split=.1)

"---------------------------------------------------"
#Evaluation


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {test_loss}, Accuracy: {test_accuracy}")

"-----------------------------------------------------"

predictions = model.predict(x_test)
predictions = (predictions > 0.5).astype(int)


"----------------------------------------------------"

# رسم دقة التدريب
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.summary()






