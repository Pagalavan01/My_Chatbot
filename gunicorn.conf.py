import os

# Render dynamically assigns a PORT environment variable
port = os.environ.get("PORT", "5000")

# Tell Gunicorn to listen on all interfaces (0.0.0.0) at the specified port
bind = f"0.0.0.0:{port}"

# Use 2 workers for free tier instances to save memory
workers = 2
