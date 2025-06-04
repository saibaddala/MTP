import socket
import numpy as np
from pynq import Overlay
from pynq.lib.dma import DMA
from pynq import allocate
import time  # For performance measurement

# Load the overlay and DMA
overlay = Overlay("Dimmer.bit")
dma = overlay.axi_dma_0

# Create a socket for the server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9092))  # Listen on all available interfaces
server_socket.listen(1)

print("Waiting for a connection...")
conn, addr = server_socket.accept()
print(f"Connected by {addr}")

# Initialize variables for performance measurement
num_images_processed = 0  # Count the number of images processed
total_fpga_processing_time = 0  # To sum up the FPGA processing times

# Start timer for total image processing
overall_start_time = time.time()  # Timer to measure total image processing time

while True:
    # Receive image dimensions first
    image_info = conn.recv(1024).decode()
    if not image_info:
        break  # Exit the loop if no data is received

    height, width, channels = map(int, image_info.split(','))

    try:
        # Allocate buffers for input and output
        input_buffer = allocate(shape=(height * width * channels,), dtype=np.uint8)
        output_buffer = allocate(shape=(height * width * channels,), dtype=np.uint8)
    except RuntimeError as e:
        print(f"Memory allocation failed: {e}")
        continue  # Skip to the next iteration or handle accordingly

    # Receive image bytes
    image_bytes = b''
    while len(image_bytes) < height * width * channels:
        packet = conn.recv(4096)
        if not packet:
            break
        image_bytes += packet

    # Copy the image bytes to the input buffer
    np.copyto(input_buffer, np.frombuffer(image_bytes, dtype=np.uint8))

    # Process the image in chunks
    rows_per_chunk = 8
    for i in range(0, height, rows_per_chunk):
        start_row = i
        end_row = min(i + rows_per_chunk, height)

        # Transfer current chunk
        dma.sendchannel.transfer(input_buffer[start_row * width * channels:end_row * width * channels])
        dma.recvchannel.transfer(output_buffer[start_row * width * channels:end_row * width * channels])

        # Wait for transfer to complete
        dma.sendchannel.wait()
        dma.recvchannel.wait()

    # Convert the output buffer to a numpy array for the processed image
    processed_image = np.reshape(output_buffer[:height * width * channels], (height, width, channels))

    # Send the processed image back to the client
    conn.sendall(output_buffer[:height * width * channels])

    # Increment image count
    num_images_processed += 1

# End timer for total image processing
overall_end_time = time.time()  
total_fpga_processing_time = overall_end_time - overall_start_time

# After all images are processed, calculate the throughput
if num_images_processed > 0:
    throughput = num_images_processed / total_fpga_processing_time  # Images per second
    print(f"Total images processed: {num_images_processed}")
    print(f"Total FPGA processing time: {total_fpga_processing_time:.6f} seconds")
    print(f"Throughput: {throughput:.2f} images per second")
else:
    print("No images were processed.")

# Close connections and destroy windows
conn.close()
server_socket.close()

