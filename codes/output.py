#function to convert image to rescaled respiratory signal

def image_to_signal(image, min_val, max_val):

    image_array = np.array(image)

    # Normalize the image array to the original signal range
    signal = (image_array / 255.0) * (max_val - min_val) + min_val

    # Normalize the signal
    rescaled_signal = ((255.0 / (signal.max() - signal.min())) * (signal - signal.min())).astype(np.uint8)

    # Rescale the image array back to the original signal range
    rescaled_signal_array = (rescaled_signal / 255.0) * (max_val - min_val) + min_val

    # Flatten the signal array
    flattened_signal_array = rescaled_signal_array.flatten()

    return flattened_signal_array



# For Fake Respiratory Signal
data_dir = '/content/drive/MyDrive/bidmc_csv'
file_name = 'bidmc_01_Signals.csv'
file_path = os.path.join(data_dir, file_name)


signal_data = pd.read_csv(file_path)
min_val = signal_data[' RESP'].min()
max_val = signal_data[' RESP'].max()


fake_image_path = './results/ppg2respiratory_gan/test_latest/images/100_fake.png'
fake_image = Image.open(fake_image_path).convert('L')

fake_flattened_signal = image_to_signal(fake_image, min_val, max_val);

#For Real Respiratory Signal

real_image_path = './results/ppg2respiratory_gan/test_latest/images/100_real.png'
real_image = Image.open(real_image_path).convert('L')

real_flattened_signal = image_to_signal(real_image, min_val, max_val);

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot for Fake Respiratory Signal
axs[0].plot(fake_flattened_signal_array)
axs[0].set_title('Fake Respiratory Signal')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')

# Plot for Real Respiratory Signal
axs[1].plot(real_flattened_signal_array)
axs[1].set_title('Real Respiratory Signal')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()
