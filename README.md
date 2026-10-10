# AI-Driven Marine Debris Detection

A web app that finds marine debris in a photo, counts each kind, and writes a short environmental report.

**Live app:** [add your Streamlit link here once it's deployed]

## How it works
1. The user uploads a photo of the water or shoreline.
2. A YOLOv8 model detects debris, scanning the whole image and then overlapping close-up sections so small items aren't missed.
3. Items are counted by type. Plastic bags, bottles, containers and waste are grouped as "plastic".
4. Google Gemini writes a short environmental report from the counts. If Gemini is unavailable, a built-in report is used instead.

## What it detects
[List your classes here, e.g. plastic, can, foam, carton, fishing net.]

## Tech used
- YOLOv8 (Ultralytics) for detection, trained on a merged marine debris dataset from Roboflow Universe
- Streamlit for the web interface
- Google Gemini API for report generation

## Run it locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"   # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```
Get a free key at https://aistudio.google.com/apikey. Never commit your key to the repo.

## Known limits
The model recognises only the classes it was trained on. It can miss very small or partly hidden items, and results are estimates, so check important findings by eye.

## Dataset credit
[Name the Roboflow datasets you used and their licenses.]
