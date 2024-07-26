from PIL import Image

#function to convert ppg and respiratory signals into image form

def save_images(data, folder_name):
    # Ensure the folder exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Iterate through each array in the data
    for i, array in enumerate(data):
        # Normalize the array to scale pixel values between 0 and 255
        rescaled = ((array - np.min(array)) / (np.max(array) - np.min(array)) * 255).astype(np.uint8)
        # Convert array to image
        img = Image.fromarray(rescaled.reshape((30, -1)), 'L')  # Reshape assumes window_size is 30, adjust if different
        # Save image
        img.save(os.path.join(folder_name, f"{i}.png"))


# Define the base directory
base_dir = 'datasets/ppg2resp'

# Save train PPG images
save_images(train_ppg_reshaped, os.path.join(base_dir, 'trainA'))
# Save test PPG images
save_images(test_ppg_reshaped, os.path.join(base_dir, 'testA'))
# Save train respiratory images
save_images(train_resp_reshaped, os.path.join(base_dir, 'trainB'))
# Save test respiratory images
save_images(test_resp_reshaped, os.path.join(base_dir, 'testB'))
