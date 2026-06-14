# Deployment Guide: Free Cloud Options

## Option 1: Railway.app (Recommended - Fastest setup, gives a nice URL)
1. Push your code to a public/private GitHub repository.
2. Go to [railway.app](https://railway.app) and sign up with GitHub.
3. Click "New Project" -> "Deploy from GitHub repo".
4. Select your FreightSense repository.
5. Railway will automatically detect the `Dockerfile` and `deploy/railway.toml`.
6. Go to the Variables tab and add:
   - `GEMINI_API_KEY`: <your-key>
   - `BHASHINI_API_KEY`: <your-key>
   - `TELEGRAM_BOT_TOKEN`: <your-bot-token>
7. Go to Networking and generate a Public Domain.
8. Your Streamlit app will be available on port 8501 (Railway maps the first port to web).

## Option 2: Hugging Face Spaces (Best for Portfolio/Interviews)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces).
2. Create new Space -> Streamlit -> Public.
3. Clone the space to your computer.
4. Copy your FreightSense project files into the HF Space folder.
5. Hugging Face only runs the Streamlit UI, so you will need to map `API_URL` to point to either localhost or bypass the FastAPI entirely inside `app.py` for a pure Streamlit deployment.
6. Add Secrets in Space Settings: `GEMINI_API_KEY`, etc.
7. Commit and push.

## Option 3: Render.com (Alternative Docker host)
1. Push to GitHub.
2. Go to [render.com](https://render.com) -> New Web Service.
3. Select your repo.
4. Environment: Docker.
5. Set env variables.
6. Note: Render free tier spins down after 15 mins of inactivity, so cold starts can take 1-2 minutes during an interview.

**Pro-tip for Placement Interviews:** Record a 2-minute video walkthrough of the app running perfectly in case the live API keys expire or the cloud free tier goes down during your demo!
