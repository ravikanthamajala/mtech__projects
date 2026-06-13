import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, GlobalAveragePooling1D, 
                                   MultiHeadAttention, LayerNormalization, Conv2D, 
                                   Reshape, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = r'D:\SRM__DS AND AI\Semester-2\DEEP LEARNING APPROACHES\Mini Project\archive (1)'
CSV_FILE = 'age_detection.csv'
MODEL_SAVE_PATH = r'D:\Python\age_prediction_model.h5'
CLASSES_SAVE_PATH = r'D:\Python\age_classes.npy'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

# Fixed Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([ 
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Fixed Model Architecture
def build_model(num_classes=4):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # EfficientNet backbone
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    x = base_model(inputs)

    # Get the proper dimensions for reshaping
    shape = x.shape
    num_patches = shape[1] * shape[2]
    projection_dim = shape[3]

    # Reshape and project patches
    x = Reshape((num_patches, projection_dim))(x)

    # Transformer block
    x = TransformerBlock(embed_dim=projection_dim, num_heads=8, ff_dim=512)(x)
    x = GlobalAveragePooling1D()(x)

    # Outputs
    age_reg = Dense(1, activation='linear', name='age_reg')(x)
    age_cls = Dense(num_classes, activation='softmax', name='age_cls')(x)

    model = Model(inputs, [age_reg, age_cls])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss={'age_reg': 'huber_loss', 'age_cls': 'categorical_crossentropy'},
        metrics={'age_cls': 'accuracy'},
        loss_weights={'age_reg': 0.4, 'age_cls': 0.6}
    )

    return model

# Get Age Group
def get_age_group(age):
    if age < 18:
        return '0-17'
    elif age < 30:
        return '18-29'
    elif age < 50:
        return '30-49'
    else:
        return '50+'

# Enhanced Data Loading with detailed debugging
def load_data():
    try:
        csv_path = os.path.join(DATASET_PATH, CSV_FILE)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"\nLoaded CSV with {len(df)} rows. First 5 entries:")
        print(df.head())

        # Convert age to numeric and clean data
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df = df.dropna(subset=['age'])
        df['age'] = df['age'].astype(int)
        df['age_group'] = df['age'].apply(get_age_group)

        le = LabelEncoder()
        df['group_label'] = le.fit_transform(df['age_group'])

        images, age_labels, group_labels = [], [], []
        missing_images = 0
        failed_images = 0

        print("\nStarting image loading...")
        for idx, row in df.iterrows():
            # Handle different path formats
            img_path = os.path.join(DATASET_PATH, row['image'])
            
            # Try alternative path joining if first attempt fails
            if not os.path.exists(img_path):
                alt_path = os.path.join(DATASET_PATH, 'images', row['image'])
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    print(f"Row {idx}: Image not found at {img_path}")
                    missing_images += 1
                    continue
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Row {idx}: Failed to load (may be corrupt) - {img_path}")
                    failed_images += 1
                    continue
                
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB format
                img = tf.keras.applications.efficientnet.preprocess_input(img)
                
                images.append(img)
                age_labels.append(row['age'])
                group_labels.append(row['group_label'])
                
                if idx % 100 == 0:
                    print(f"Processed {idx+1}/{len(df)} images")
                    
            except Exception as e:
                print(f"Row {idx}: Error processing image - {str(e)}")
                failed_images += 1

        print("\nImage loading summary:")
        print(f"Total CSV rows: {len(df)}")
        print(f"Successfully loaded images: {len(images)}")
        print(f"Missing images: {missing_images}")
        print(f"Failed/corrupt images: {failed_images}")

        if len(images) == 0:
            raise ValueError("No valid images were loaded. Check your dataset paths and integrity.")

        X = np.array(images)
        y_reg = np.array(age_labels)
        y_cls = tf.keras.utils.to_categorical(group_labels, num_classes=len(le.classes_))

        return X, y_reg, y_cls, le.classes_

    except Exception as e:
        print(f"\nError in load_data(): {str(e)}")
        raise

# Enhanced Training Pipeline
def train_model():
    try:
        print("\nStarting training process...")
        X, y_reg, y_cls, classes = load_data()

        # Data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.15
        )

        # Create training and validation generators
        train_generator = datagen.flow(
            X,
            {'age_reg': y_reg, 'age_cls': y_cls},
            batch_size=BATCH_SIZE,
            subset='training'
        )

        validation_generator = datagen.flow(
            X,
            {'age_reg': y_reg, 'age_cls': y_cls},
            batch_size=BATCH_SIZE,
            subset='validation'
        )

        model = build_model(num_classes=len(classes))
        print("\nModel summary:")
        model.summary()

        print("\nTraining model...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_age_cls_accuracy',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    MODEL_SAVE_PATH,
                    save_best_only=True,
                    monitor='val_age_cls_accuracy'
                )
            ],
            verbose=1
        )

        # Save model and classes
        model.save(MODEL_SAVE_PATH)
        np.save(CLASSES_SAVE_PATH, classes)
        print(f"\nModel successfully saved to {MODEL_SAVE_PATH}")
        print(f"Class labels saved to {CLASSES_SAVE_PATH}")

        return model, classes

    except Exception as e:
        print(f"\nError in train_model(): {str(e)}")
        raise

# Enhanced Prediction Class
class AgePredictor:
    def __init__(self, force_train=False):
        self.detector = MTCNN()
        
        if force_train or not os.path.exists(MODEL_SAVE_PATH):
            print("\nModel not found or forced training - starting training...")
            self.model, self.classes = train_model()
        else:
            try:
                print("\nLoading existing model...")
                self.model = tf.keras.models.load_model(
                    MODEL_SAVE_PATH,
                    custom_objects={'TransformerBlock': TransformerBlock}
                )
                self.classes = np.load(CLASSES_SAVE_PATH)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Falling back to training new model...")
                self.model, self.classes = train_model()

    def predict(self, frame):
        try:
            # Convert to RGB and detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(rgb_frame)
            
            if not results:
                return frame

            # Get largest face
            main_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = main_face['box']
            
            # Ensure coordinates are within frame bounds
            x, y = max(0, x), max(0, y)
            w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

            # Extract and preprocess face
            face = rgb_frame[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = tf.keras.applications.efficientnet.preprocess_input(face)

            # Make prediction
            age, group = self.model.predict(face[np.newaxis, ...], verbose=0)
            predicted_age = max(0, int(age[0][0]))  # Ensure age isn't negative
            age_group = self.classes[np.argmax(group[0])]
            confidence = np.max(group[0])

            # Draw results on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"Age: {predicted_age} ({age_group}, {confidence:.0%})"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            print(f"Prediction error: {str(e)}")

        return frame

# Main Execution with better error handling
if __name__ == "__main__":
    try:
        print("\nInitializing Age Prediction System...")
        
        # First test dataset structure
        print("\nTesting dataset structure...")
        csv_path = os.path.join(DATASET_PATH, CSV_FILE)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            test_img_path = os.path.join(DATASET_PATH, df.iloc[0]['image'])
            print(f"Sample image path: {test_img_path}")
            print(f"Exists: {os.path.exists(test_img_path)}")
        else:
            print(f"CSV file not found at {csv_path}")

        # Initialize predictor
        predictor = AgePredictor(force_train=False)

        # Start real-time prediction
        print("\nStarting real-time prediction...")
        print("Press 'q' to quit")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open video capture")

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            frame = predictor.predict(frame)
            cv2.imshow('Age Prediction', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\nApplication closed successfully")

    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()