import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# Define the folder structure and data directories
train_dir = 'G:\Bachelor_Project\\revised_segmentedTF\\training'
val_dir = 'G:\Bachelor_Project\\revised_segmentedTF\\validating'

# Data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

# Load and preprocess the data
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(45, 100),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(45, 100),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define a function to create the CNN model with tunable hyperparameters
def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('conv1_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu', input_shape=(45, 100, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=128, max_value=1024, step=128), activation='relu'))
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(2, activation='softmax'))

    # Compile the model with hyperparameters
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Implement hyperparameter search using Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='my_tuning_dir',
    project_name='my_tuning_project'
)

# Perform the hyperparameter search
tuner.search(train_generator, epochs=5, validation_data=val_generator)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and compile the final model with the best hyperparameters
final_model = build_model(best_hps)

# Train the final model with more epochs
num_epochs = 20  # Increase the number of epochs for the final training
history = final_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Output the full model summary
final_model.summary()

# Predict the labels using your trained model
predicted_labels = model.predict(val_generator)

# Convert one-hot encoded labels to integers
predicted_classes = np.argmax(predicted_labels, axis=1)
true_classes = val_generator.classes 

# Compute the confusion matrix
confusion = confusion_matrix(true_classes, predicted_classes)

print("Confusion Matrix:")
print(confusion)

# Output a classification report
report = classification_report(true_classes, predicted_classes)
print("Classification Report:")
print(report)