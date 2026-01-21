# generate_zerodha_token.py

import yaml
from kiteconnect import KiteConnect
import webbrowser
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# Global variable to capture the token from the redirect
captured_request_token = None

class RedirectHandler(BaseHTTPRequestHandler):
    """
    A custom HTTP request handler to capture the redirect from Zerodha.
    """
    def do_GET(self):
        global captured_request_token
        
        query_components = parse_qs(urlparse(self.path).query)
        request_token = query_components.get("request_token", [None])[0]

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        if request_token:
            captured_request_token = request_token
            self.wfile.write(b"<html><body><h1>Authentication successful!</h1>"
                             b"<p>You can close this browser tab now.</p></body></html>")
            print("\nSuccessfully captured request_token.")
        else:
            self.wfile.write(b"<html><body><h1>Authentication failed.</h1>"
                             b"<p>Request token not found in URL. Please try again.</p></body></html>")
            print("\nFailed to capture request_token.")
        
        # Shut down the server in a separate thread to avoid deadlocks
        threading.Thread(target=self.server.shutdown).start()

def generate_token():
    """
    A one-time script to generate the Zerodha access token automatically.
    """
    # --- 1. Load Configuration ---
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / 'config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['zerodha_credentials']

    api_key = config['api_key']
    api_secret = config['api_secret']
    redirect_url = config['redirect_url']

    
    # Initialize the Kite client
    kite = KiteConnect(api_key=api_key)

    # --- Parse port from the URL ---
    try:
        port = urlparse(redirect_url).port
        if not port:
            raise ValueError("Port not found in redirect_url. Please ensure it is correct (e.g., http://127.0.0.1:8080).")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return


    # --- 2. Start Local Web Server ---
    port = 8080 # A common port, ensure it's free
    server = HTTPServer(('', port), RedirectHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"Started temporary web server on port {port}...")

    # --- 3. Generate and Open Login URL ---
    login_url = kite.login_url()
    print(f"\nYour browser will now open for Zerodha login...")
    webbrowser.open(login_url)

    # Wait for the server to capture the token (with a timeout)
    server_thread.join(timeout=120)
    server.server_close()

    if not captured_request_token:
        print("\nOperation timed out or failed. Please run the script again.")
        return

    # --- 4. Generate and Save the Access Token ---
    try:
        session = kite.generate_session(captured_request_token, api_secret=api_secret)
        access_token = session.get("access_token")

        if not access_token:
            print("\nFailed to generate access token from request token. Response:")
            print(session)
            return

        print(f"\nSuccessfully generated access token.")

        local_config_path = project_root / 'local_config.yml'

        try:
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            local_config = {}

        # Safely update the config dictionary
        if 'zerodha' not in local_config:
            local_config['zerodha'] = {}
        local_config['zerodha']['access_token'] = access_token
        
        with open(local_config_path, 'w') as f:
            yaml.dump(local_config, f, default_flow_style=False)
            
        print(f"Access token safely updated in '{local_config_path}'.")

    except Exception as e:
        print(f"\nAn error occurred during token generation: {e}")

if __name__ == "__main__":
    generate_token()