import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import os

# ==================================================================
# Configuration (Update these paths according to your system)
# ==================================================================
DATASET_PATH = r'D:\SRM__DS AND AI\Semester-2\DEEP LEARNING APPROACHES\Mini Project\archive (1)'
CSV_FILE = 'age_detection.csv'
MODEL_SAVE_PATH = r'D:\Python\age_detection_model.h5'
LABEL_SAVE_PATH = r'D:\Python\age_group_classes.npy'

# ==================================================================
# Step 1: Check for existing model or train new one
# ==================================================================
def train_model():
    # Load dataset
    df = pd.read_csv(os.path.join(DATASET_PATH, CSV_FILE))
    
    # Preprocessing
    def convert_age(age_str):
        try: return int(age_str.split('-')[0])
        except: return np.nan
    
    df['age_num'] = df['age'].apply(convert_age).dropna().astype(int)
    df['age_group'] = df['age_num'].apply(lambda x: '18-25' if 18<=x<=25 else
                                                 '26-35' if 26<=x<=35 else
                                                 '36-45' if 36<=x<=45 else '46-60')
    
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['age_group'])
    
    # Load images
    images = []
    labels = []
    for idx, row in df.iterrows():
        try:
            img_path = os.path.join(DATASET_PATH, row['file'])
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(row['label'])
        except:
            continue
    
    # Convert to arrays
    X = np.array(images) / 255.0
    y = pd.get_dummies(labels).values
    
    # Build model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, validation_split=0.2)
    
    # Save assets
    model.save(MODEL_SAVE_PATH)
    np.save(LABEL_SAVE_PATH, le.classes_)
    return model, le.classes_

# ==================================================================
# Step 2: Face detection and age prediction
# ==================================================================
def main():
    # Try to load existing model
    try:
        model = load_model(MODEL_SAVE_PATH)
        age_groups = np.load(LABEL_SAVE_PATH)
        print("Loaded pre-trained model")
    except:
        print("Training new model...")
        model, age_groups = train_model()
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            try:
                face_img = frame[y:y+h, x:x+w]
                resized = cv2.resize(face_img, (128,128))
                normalized = resized.astype('float32') / 255.0
                prediction = model.predict(normalized[np.newaxis, ...])[0]
                
                age_group = age_groups[np.argmax(prediction)]
                confidence = np.max(prediction)
                
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"{age_group} ({confidence:.0%})",
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            except Exception as e:
                print(f"Face processing error: {e}")
        
        cv2.imshow('Age Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()