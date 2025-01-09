This repository provides a FastAPI endpoint for interacting with the f5-tts text-to-speech model hosted locally.

Installation

Clone the repository:

Bash

git clone https://github.com/your-username/f5-tts-api.git
cd f5-tts-api
Install dependencies:

Bash

pip install -r requirements.txt
Usage

Start the API:

Bash

uvicorn main:app --reload
This will start the FastAPI server on http://127.0.0.1:8000 (by default).