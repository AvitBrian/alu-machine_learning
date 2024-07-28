import os
import subprocess
import sys

def install_requirements():
    """Install the required packages from requirements.txt."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Successfully installed requirements.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing requirements: {e}")
        sys.exit(1)

def run_app():
    """Run the Flask app using Gunicorn."""
    try:
        subprocess.check_call(['gunicorn', 'app:app', '--bind', '0.0.0.0:8000', '--timeout', '120'])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while starting the app: {e}")
        sys.exit(1)

if __name__ == '__main__':
    install_requirements()
    run_app()
