#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm


image_dir=r"C:\Users\anjan\Downloads\Dataset_Celebrities\cropped"
virat_images=os.listdir(image_dir+ '\\virat_kohli')
serena_images=os.listdir(image_dir+ '\\serena_williams')
maria_images=os.listdir(image_dir+ '\\maria_sharapova')
roger_images=os.listdir(image_dir+ '\\roger_federer')
lionel_images=os.listdir(image_dir+ '\\lionel_messi')


print("--------------------------------------\n")

print('The length of Virat Kohli images is',len(virat_images))
print('The length of Serena Williams images is',len(serena_images))
print('The length of Maria Sharapova images is',len(maria_images))
print('The length of Rodger Federer images is',len(roger_images))
print('The length of Lionel Messi images is',len(lionel_images))



print("--------------------------------------\n")
dataset=[]
label=[]
img_siz=(128,128)


for i , image_name in tqdm(enumerate(virat_images),desc="Virat Kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(serena_images),desc="Serena Williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)
        
for i , image_name in tqdm(enumerate(roger_images),desc="Roger Federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)
        
        
for i ,image_name in tqdm(enumerate(maria_images),desc="Maria Sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)        
        
for i ,image_name in tqdm(enumerate(lionel_images),desc="Lionel Messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)        
              
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.3,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")


# Normalizing the Dataset
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Correcting the labels if needed (Ensure they range from 0 to 4 for 5 classes)
# label = label - 1 (If labels don't start from 0)

# Build the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # Change to 5 neurons for 5 classes
])

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Change loss function
              metrics=['accuracy'])

# Training the Model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.3)

# Model Evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {round(accuracy * 100, 2)}')

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report:\n', classification_report(y_test, y_pred))


# Load and preprocess a single image
def preprocess_single_image(image_path):
    img_size = (128, 128)
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize(img_size)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    return image

image_path_to_predict = [r"C:\Users\anjan\Downloads\Dataset_Celebrities\cropped\virat_kohli\virat_kohli5.png",
                         r"C:\Users\anjan\Downloads\Dataset_Celebrities\cropped\serena_williams\serena_williams4.png",
                         r"C:\Users\anjan\Downloads\Dataset_Celebrities\cropped\maria_sharapova\maria_sharapova6.png",
                         r"C:\Users\anjan\Downloads\Dataset_Celebrities\cropped\roger_federer\roger_federer4.png",
                         r"C:\Users\anjan\Downloads\Dataset_Celebrities\cropped\lionel_messi\lionel_messi8.png"]

# Preprocess the single image
for i in image_path_to_predict:
    single_image = preprocess_single_image(i)

    # Reshape the image to fit the model's input shape
    single_image = np.expand_dims(single_image, axis=0)

    # Make predictions using the model
    predictions = model.predict(single_image)
    predicted_class = np.argmax(predictions)

    class_names = ['Virat Kohli', 'Serena Williams', 'Roger Federer', 'Maria Sharapova', 'Lionel Messi']
    predicted_label = class_names[predicted_class]

    print(f"The predicted label for the image is: {predicted_label}")


# DOCUMENTATION

# The chosen model architecture is Simple CNN Model architecture. Among different type of models, CNN has high performance on image classification.
# During the data preprocessing phase, we use 5 different for loops to iterate through each palyer, resize it and append to the dataset along with 
# the respective label. The dataset is converted to numpy arrays which is compactible with Tensorflow.
# Then we normalize it and build the CNN model.

# During the Model training, we fit the CNN model to the training dataset and set the epochs as 50 and batch_size as 32. We set the validation split as 
# 0.3, i.e; it takes 30% of the training data as validation set. So that it helps to prevent the model from overfitting.

# To understand the model's performance, we print the classification report which shows the precision, recall, F1-score, and accuracy of each class.
# Here, the accuracy is 84.31%. The predictions for each class is also printed.