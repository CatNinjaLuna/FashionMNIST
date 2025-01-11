import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Load an image (replace 'image_path' with the actual image path)
image_path = 'selfieCat.png'  # Specify your image path
img = load_img(image_path)  # Load image
img_array = img_to_array(img)  # Convert image to numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Define ImageDataGenerator with augmentations
datagen = ImageDataGenerator(
    rotation_range=40,         # Random rotations from -40 to 40 degrees
    width_shift_range=0.2,     # Random horizontal shifts
    height_shift_range=0.2,    # Random vertical shifts
    zoom_range=0.2            # Random zoom
)

# Fit the datagen on the image (required before using .flow())
datagen.fit(img_array)

# Generate augmented images
augmented_images = datagen.flow(img_array, batch_size=1)

# Plot original and augmented images in one row
plt.figure(figsize=(16, 4))

# Main title above all subplots
plt.suptitle('Data Augmentation on Example Images', fontsize=13, y=0.99)

# Original image
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Generate and display augmented images with specific titles
for i in range(2, 5):  # Displaying only 4 images (including original)
    augmented_img = next(augmented_images)[0].astype('uint8')
    
    plt.subplot(1, 4, i)  # Adjusted to 1x4 grid (1 row, 4 columns)
    
    if i == 2:
        # Zoom In/Out - Zoom range is applied randomly, so we can't display specific values directly.
        plt.title('Zoom In/Out')
    elif i == 3:
        # Rotation - Rotation is randomly selected within the range of -40 to 40 degrees
        rotation_value = np.random.randint(-40, 40)  # Example: Rotation by X degrees
        plt.title(f'Rotation by {rotation_value}°')
    elif i == 4:
        # Height Shift - Height shift is randomly applied within 0.2 of image height
        height_shift_value = np.random.uniform(-0.2, 0.2)  # Example: Height shift by Y percent
        plt.title(f'Height Shift by {height_shift_value*100:.1f}%')
    
    plt.imshow(augmented_img)
    plt.axis('off')

plt.tight_layout()
plt.show()
