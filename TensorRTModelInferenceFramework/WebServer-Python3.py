from http.server import BaseHTTPRequestHandler, HTTPServer

# HTTPRequestHandler class
class ImageRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            bits = bytes(load_binary('/tmp/Snapshot.png'))
            self.send_header('Content-Length', str(len(bits)))
            self.end_headers()
            self.wfile.write(bits)
            print('get complete')

def load_binary(file):
    print('load binary file')
    with open(file, 'rb') as file:
        return file.read()


def run_server():
    print('Web Server Started')

    server_address = ('127.0.0.1', 8080)

    httpd = HTTPServer(server_address, ImageRequestHandler)

    print('running server...')
    httpd.serve_forever()

run_server()
