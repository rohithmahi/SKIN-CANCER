import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt  
import pandas as pd

# Define image height and width
img_height = 180
img_width = 180

# Load the saved model
model = load_model('skin_cancer_inceptionv3.keras')

# Define class names
class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 
               'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

# Define a dictionary mapping each class to its corresponding Excel file
excel_files = {
    'actinic keratosis': 'keratosis.xlsx',
    'basal cell carcinoma': 'basalcall.xlsx',
    'dermatofibroma': 'Dermatofibroma.xlsx',
    'melanoma': 'melanoma.xlsx',
    'nevus': 'nevus.xlsx',
    'pigmented benign keratosis': 'benign.xlsx',
    'seborrheic keratosis': 'keratosis.xlsx',
    'squamous cell carcinoma': 'squamous.xlsx',
    'vascular lesion': 'vascular.xlsx'
}

# Define a function to preprocess the image
def preprocess_image(img_path, img_height, img_width):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return x

# Load the image file
img_path = r"D:\PROJECTS\Rohith\skin cancer cnn\testing\ISIC_0011865.jpg"

# Preprocess the image
x = preprocess_image(img_path, img_height, img_width)

# Make predictions on the preprocessed image
preds = model.predict(x)

# Get the class with the highest probability
pred_class_index = np.argmax(preds[0])
pred_class = class_names[pred_class_index]

# Display the image
plt.imshow(image.load_img(img_path))
plt.axis('off')  # Turn off axis
plt.title(f'Predicted Class: {pred_class}')
plt.show()

# Load recommendations from the corresponding Excel file
excel_filename = excel_files[pred_class]
df = pd.read_excel(excel_filename)

print("Recommendation for", pred_class)
print(df)