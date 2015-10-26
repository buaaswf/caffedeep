import SocketServer
from SocketServer import StreamRequestHandler as SRH
from time import ctime
from time import sleep
import numpy as np
import featureservice
import sys
import Image

host = 'localhost'
port = 3201
addr = (host, port)

RECV_SIZE = 1024

class Servers(SRH):
    def handle(self):
        print 'Got connection from ',self.client_address
       # self.wfile.write('Connection %s:%s at %s succeed!' % (host, port, ctime()))
        #while True:
        try:
            rev_data = self.request.recv(RECV_SIZE)
        #self.wfile.write('Recv: %s' % rev_data)
        except Exception as e:
            print str(e)
        dirs=rev_data.split(";")
        featureservice.image2mat(dirs[0],dirs[1],"/home/s.li/caffedeep/caffe-master/examples/generatemat/serverres.mat")
        self.request.sendall("/home/s.li/caffedeep/caffe-master/examples/generatemat/serverres.mat")
if __name__ == "__main__":
    print 'Server is running....'
    server = SocketServer.ThreadingTCPServer(addr,Servers)
    server.serve_forever()

