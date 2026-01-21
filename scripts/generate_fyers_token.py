import yaml
from fyers_apiv3 import fyersModel
import webbrowser
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# This is a simple approach for a single-purpose script.
captured_auth_code = None

class RedirectHandler(BaseHTTPRequestHandler):
    """
    A custom HTTP request handler to capture the redirect from Fyers.
    """
    def do_GET(self):
        global captured_auth_code
        
        query_components = parse_qs(urlparse(self.path).query)
        auth_code = query_components.get("auth_code", [None])[0]

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        if auth_code:
            captured_auth_code = auth_code
            self.wfile.write(b"<html><body><h1>Authentication successful!</h1>"
                             b"<p>You can close this browser tab now.</p></body></html>")
            print("\nSuccessfully captured auth code.")
        else:
            self.wfile.write(b"<html><body><h1>Authentication failed.</h1>"
                             b"<p>Auth code not found in URL. Please try again.</p></body></html>")
            print("\nFailed to capture auth code.")
        
        threading.Thread(target=self.server.shutdown).start()

def generate_token():
    """
    A one-time script to generate the Fyers access token automatically
    by running a temporary local web server to catch the redirect.
    """
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / 'config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['fyers_credentials']

    client_id = config['client_id']
    secret_key = config['secret_key']
    redirect_url = config['redirect_url']
    port = int(urlparse(redirect_url).port)

    server = HTTPServer(('', port), RedirectHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"Started temporary web server on port {port}...")

    # --> 1. The SessionModel object is created and assigned to 'session' here
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_url,
        response_type='code',
        grant_type='authorization_code'
    )

    auth_url = session.generate_authcode()
    print(f"\nYour browser will now open for Fyers login...")
    webbrowser.open(auth_url)

    server_thread.join(timeout=120)
    server.server_close()

    if not captured_auth_code:
        print("\nOperation timed out or failed. Please run the script again.")
        return

    # --> 2. The 'session' object is used here to set the token...
    session.set_token(captured_auth_code)
    # --> 3. ...and here to generate the final access token.
    response = session.generate_token()

    if response.get("access_token"):
        access_token = response["access_token"]
        print(f"\nSuccessfully generated access token.")

        local_config_path = project_root / 'local_config.yml'

        try:
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            local_config = {}

        if 'fyers_credentials' not in local_config:
            local_config['fyers_credentials'] = {}
        local_config['fyers_credentials']['access_token'] = access_token
        
        with open(local_config_path, 'w') as f:
            yaml.dump(local_config, f)
            
        print(f"Access token safely updated in '{local_config_path}'.")
    else:
        print("\nFailed to generate access token from auth code. Response:")
        print(response)

if __name__ == "__main__":
    generate_token()