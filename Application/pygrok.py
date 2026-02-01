from pyngrok import ngrok
import subprocess
import time
import threading
import os
import sys # Import sys to get the Python executable

# Terminate any existing ngrok tunnels
ngrok.kill()

# Authenticate ngrok (replace "YOUR_NGROK_AUTH_TOKEN" with your actual token)
# You can get a free auth token from https://ngrok.com/signup
ngrok.set_auth_token("38hIznCEgbXQmfBIaBBb8EZ7V6S_6T4m2iu1y8PY5FF145yTd") # <<< REPLACE WITH YOUR NGROK AUTH TOKEN

port = 8501

def run_streamlit():
    # Ensure Streamlit runs from the correct directory if needed, or specify absolute path to app
    streamlit_path = os.path.join(os.getcwd(), "streamlit_app.py")
    # Use sys.executable to ensure the correct Python interpreter is used
    subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_path, "--server.port", str(port), "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"])

# Run Streamlit in a separate thread
streamlit_thread = threading.Thread(target=run_streamlit)
streamlit_thread.daemon = True # Daemonize thread to exit when main program exits
streamlit_thread.start()

# Give Streamlit some time to start up
time.sleep(15) # Increased sleep time for better reliability

# Open a ngrok tunnel to the Streamlit app
try:
    tunnel = ngrok.connect(port)
    print(f"Streamlit App URL: {tunnel.public_url}")
except Exception as e:
    print(f"Error creating ngrok tunnel: {e}")
    print("Please ensure your ngrok authentication token is correctly set and try again.")