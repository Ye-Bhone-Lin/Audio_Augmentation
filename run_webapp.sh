#!/bin/bash

# Navigate to the project root directory (optional, but good practice)
# cd /Users/htutkoko/Study Data/LU Lab/ASR_Augmentation

echo "Activating pipenv shell and starting Flask web application..."

# Activate the pipenv shell and run the Flask app
pipenv run python web_app/app.py

echo "Web application started. Access it at http://127.0.0.1:5000"
echo "Press Ctrl+C in this terminal to stop the application."
