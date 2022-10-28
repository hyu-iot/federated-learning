# https://www.thepythoncode.com/article/send-receive-files-using-sockets-python

import socket
import tqdm
import os

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096  # send 4096 bytes each time step


# The ip address or hostname of the server, the receiver
host = "127.0.0.1"

# The port
port = 5001

# The name of file we want to send
filename = "model.pth"

# Get the file size
filesize = os.path.getsize(filename)

# Create the client socket
s = socket.socket()

# Connecting to the server
print(f"[+] Connecting to {host}:{port}")
s.connect((host, port))
print("[+] Connected.")

# Send the filename and filesize
s.send(f"{filename}{SEPARATOR}{filesize}".encode())

# Start sending the file.
progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
with open(filename, "rb") as f:
    while True:
        # Read the bytes from the file.
        bytes_read = f.read(BUFFER_SIZE)
        if not bytes_read:
            # File transmitting is done.
            break
        # We use sendall to assure transmission in busy networks
        s.sendall(bytes_read)
        # update the progress bar
        progress.update(len(bytes_read))

# Close the socket
s.close()