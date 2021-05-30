heroku ps:scale web=1
web: uvicorn starter.main:app --host=0.0.0.0 --port=${PORT:-8000}
