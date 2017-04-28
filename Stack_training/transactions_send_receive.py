# Script for sending and receiving stored transactions from playing
# consider the file "Test_Data.npy"

import socket                   # Import socket module

# Local Game Playing End
def send_transactions():
	port = 60000                    # Reserve a port for your service.
	s = socket.socket()             # Create a socket object
	host = socket.gethostname()     # Get local machine name
	s.bind((host, port))            # Bind to the port
	s.listen(5)                     # Now wait for client connection.

    # waiting for the remote GPU server to request transactions
	while True:
	    conn, addr = s.accept()     # Establish connection with client.
	    print 'Got connection from', addr
	    data = conn.recv(1024)
	    print('Server received', repr(data))

	    filename='Test_Data.npy'
	    f = open(filename,'rb')
	    l = f.read(1024)
	    while (l):
	       conn.send(l)
	       print('Sent ',repr(l))
	       l = f.read(1024)
	    f.close()

	    conn.close()


# Remote GPU host end
def receive_transactions():

	s = socket.socket()             # Create a socket object
	host = socket.gethostname()     # Get local machine name
	port = 60000                    # Reserve a port for your service.

	s.connect((host, port))

	with open('Test_Data.npy', 'wb') as f:
	    print 'file opened'
	    while True:
	        print('receiving data...')
	        data = s.recv(1024)
	        if not data:
	            break
	        # write data to a file
	        f.write(data)

	f.close()
	print('Successfully get the file')
	s.close()
	print('connection closed')
