import numpy as np
from PIL import Image

def reversible_data_hiding_rgba(data, cover_image):
    # Convert the data to binary
    binary_data = ''.join(format(ord(char), '08b') for char in data)

    # Load the cover image
    cover_array = np.array(cover_image)

    # Split the RGBA channels
    channels = [cover_array[:, :, i] for i in range(4)]

    # Get the image dimensions
    height, width = channels[0].shape

    # Calculate the maximum embedding capacity
    max_capacity = height * width * 4

    # Check if the data can be embedded within the cover image
    if len(binary_data) > max_capacity:
        raise ValueError("Insufficient space in the cover image to embed the data.")

    # Embed the binary data in all channels
    data_index = 0
    for channel in channels:
        # Find the histogram of the channel
        histogram = np.bincount(channel.flatten(), minlength=256)

        # Find the zero point and peak point in the histogram
        zero_point = np.argmin(histogram)
        peak_point = np.argmax(histogram)

        # Shift the histogram to the right
        channel[channel > zero_point] += 1

        # Embed the binary data in the channel
        for i in range(height):
            for j in range(width):
                if channel[i, j] == peak_point:
                    if data_index < len(binary_data):
                        if binary_data[data_index] == '1':
                            channel[i, j] += 1
                        data_index += 1

    # Combine the modified channels
    marked_array = np.stack(channels, axis=2)

    # Create the marked image
    marked_image = Image.fromarray(marked_array)

    return marked_image

def extract_data_rgba(marked_image_path):
    # Load the marked image
    marked_image = Image.open(marked_image_path)
    marked_array = np.array(marked_image)

    # Split the RGBA channels
    channels = [marked_array[:, :, i] for i in range(4)]

    # Get the image dimensions
    height, width = channels[0].shape

    # Extract the binary data from all channels
    binary_data = ''
    for channel in channels:
        # Find the histogram of the channel
        histogram = np.bincount(channel.flatten(), minlength=256)

        # Find the zero point and peak point in the histogram
        zero_point = np.argmin(histogram)
        peak_point = np.argmax(histogram)

        # Extract the binary data from the channel
        for i in range(height):
            for j in range(width):
                if channel[i, j] == peak_point + 1:
                    binary_data += '1'
                elif channel[i, j] == peak_point:
                    binary_data += '0'

    # Convert the binary data to ASCII characters
    data = ''
    for i in range(0, len(binary_data), 8):
        char = chr(int(binary_data[i:i+8], 2))
        data += char

    return data