import socket
import pickle

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the IP address and port
server_socket.bind(('0.0.0.0', 12345))  # Listen on all available interfaces, port 12345

# Start listening for incoming connections
server_socket.listen(1)
print('Waiting for a connection...')

# Wait for a connection
client_socket, client_address = server_socket.accept()
print(f'Connection from {client_address}')

# Receive data
data = client_socket.recv(1024)  # Receive up to 1024 bytes of data
integers = pickle.loads(data) 
print(f'Received data: {integers}')

# Send a response (optional)
client_socket.sendall(b'Data received')

# Close the connection
# client_socket.close()
