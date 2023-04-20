import zmq
from http.server import BaseHTTPRequestHandler, HTTPServer

game_server = 'tcp://localhost:8888'

http_hostname = 'localhost'
http_port = 8080

context = zmq.Context()
game_server_socket = context.socket(zmq.REQ)
game_server_socket.connect(game_server)

class StatsServer(BaseHTTPRequestHandler):
    def do_GET(self):
        req = {
            'method': 'stats'
        }
        game_server_socket.send_json(req)
        res = game_server_socket.recv()
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(res)

if __name__ == "__main__":        
    proxy = HTTPServer((http_hostname, http_port), StatsServer)
    print("Server started http://%s:%s" % (http_hostname, http_port))

    try:
        proxy.serve_forever()
    except KeyboardInterrupt:
        pass

    proxy.server_close()
