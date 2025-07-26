🎭 Deepfake Detection with VGG19 A Deep Learning Approach to Exposing the Fake

Welcome to a project focused on one of the most critical challenges of our digital age — detecting deepfakes. Using transfer learning with VGG19, this model is trained to classify face images as either real or synthetically generated.

⚠️ Whether it’s safeguarding media integrity or stopping misinformation, this model has your back.

📌 Key Features 🔍 Binary Image Classification: Real vs. Fake

🧠 VGG19 pre-trained on ImageNet for powerful feature extraction

🚀 Fine-tuned classifier with high accuracy and minimal overfitting

📊 Evaluation tools including confusion matrix, F1-score, and ROC-AUC

🎯 Single image prediction with confidence score

🗂️ Dataset Structure Compatible with any image dataset following this format: dataset/ ├── train/ │ ├── real/ │ └── fake/ ├── val/ │ ├── real/ │ └── fake/ └── test/ ├── real/ └── fake/ You can use datasets like:

FaceForensics++

Deepfake Detection Challenge (DFDC)

Or your own custom dataset

🧬 Model Architecture Built on VGG19 for feature extraction:

🧠 VGG19(weights='imagenet', include_top=False)

🔄 GlobalAveragePooling2D()

🧱 Dense(128, activation='relu')

🧪 Dropout(0.5)

🎯 Dense(1, activation='sigmoid')

Only the top layers are trained — the base remains frozen to prevent overfitting and speed up training.

⚙️ Setup Guide Clone the Repository

(Optional) Create and activate a virtual environment

Install Dependencies tensorflow
numpy
matplotlib
scikit-learn
opencv-python

Run: pip install -r requirements.txt

🚀 Training Train the model with: python train.py

You can tweak parameters like:

epochs

batch_size

image_size

learning_rate

All configurable in train.py. 🧪 Evaluation 📊 Metrics:

Accuracy

Precision / Recall / F1-Score

Confusion Matrix

ROC-AUC

🔍 Image Prediction Make predictions on individual images: python predict.py --image path/to/image.jpg

Example result:

Prediction: FAKE (Confidence: 94.12%)
