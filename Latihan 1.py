import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Latihan 1: CNN dari Scratch untuk CIFAR-10
def praktikum_cnn_from_scratch():
    print("LATIHAN 1: CNN DARI AWAL UNTUK CIFAR-10")
    print("=" * 50)
    
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # One-hot encode labels
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # Visualize sample images
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_train[i])
        plt.title(class_names[np.argmax(y_train[i])])
        plt.axis('off')
    plt.suptitle("Sample Images from CIFAR-10", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Build CNN from scratch
    def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
        model = keras.Sequential([
            # Convolutional Block 1
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                         input_shape=input_shape, name='conv1'),
            layers.BatchNormalization(name='bn1'),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2'),
            layers.BatchNormalization(name='bn2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            # Convolutional Block 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3'),
            layers.BatchNormalization(name='bn3'),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv4'),
            layers.BatchNormalization(name='bn4'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Convolutional Block 3
            layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv5'),
            layers.BatchNormalization(name='bn5'),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv6'),
            layers.BatchNormalization(name='bn6'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25, name='dropout3'),
            
            # Fully Connected Layers
            layers.Flatten(name='flatten'),
            layers.Dense(256, activation='relu', name='fc1'),
            layers.BatchNormalization(name='bn7'),
            layers.Dropout(0.5, name='dropout4'),
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        return model
    
    # Create model
    model = build_cnn_model()
    
    # Display model architecture
    print("\nCNN MODEL ARCHITECTURE:")
    print("=" * 40)
    model.summary()
    
    # Visualize model architecture
    keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("\nTRAINING CNN FROM SCRATCH...")
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=50,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize predictions
    y_pred = model.predict(X_test[:25])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test[:25], axis=1)
    
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_test[i])
        
        # Color code predictions
        color = 'green' if y_pred_classes[i] == y_true_classes[i] else 'red'
        plt.title(f"True: {class_names[y_true_classes[i]]}\nPred: {class_names[y_pred_classes[i]]}", 
                 color=color, fontsize=8)
        plt.axis('off')
    
    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Visualize feature maps from first convolutional layer
    print("\nVISUALIZING FEATURE MAPS...")
    layer_outputs = [layer.output for layer in model.layers[:8]]  # First 8 layers
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations for a sample image
    sample_image = X_test[0:1]
    activations = activation_model.predict(sample_image)
    
    # Visualize feature maps
    layer_names = [layer.name for layer in model.layers[:8]]
    
    for layer_name, layer_activation in zip(layer_names, activations):
        if 'conv' in layer_name:
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            
            n_cols = 8
            n_rows = n_features // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
            fig.suptitle(f'Feature Maps: {layer_name}', fontsize=16)
            
            for i in range(n_features):
                ax = axes[i // n_cols, i % n_cols]
                ax.imshow(layer_activation[0, :, :, i], cmap='viridis')
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
            break  # Show only first conv layer
    
    return model, history, test_accuracy

# Jalankan latihan 1
cnn_model, training_history, cnn_accuracy = praktikum_cnn_from_scratch()