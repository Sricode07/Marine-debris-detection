"""
Marine Debris Detection + Environmental Report

YOLO detects marine debris.
Gemini generates an environmental report when available.
If Gemini is unavailable, a local fallback report is generated.
"""

import argparse
import os
from collections import Counter

from google import genai
from ultralytics import YOLO


# Gemini model
GEMINI_MODEL = "gemini-3.1-flash-lite"


# ---------------------------------------------------------
# YOLO DETECTION
# ---------------------------------------------------------

def run_detection(weights_path, image_path, conf_threshold=0.35):

    model = YOLO(weights_path)

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True
    )

    result = results[0]

    class_names = result.names

    detected_classes = [
        class_names[int(box.cls)]
        for box in result.boxes
    ]

    counts = Counter(detected_classes)

    print(f"Annotated image saved under: {result.save_dir}")

    return counts


# ---------------------------------------------------------
# GEMINI PROMPT
# ---------------------------------------------------------

def build_prompt(counts):

    if not counts:

        return """
An underwater/coastal image was analyzed using an AI vision model,
but no marine debris was detected.

Write a short environmental report with:

**Pollution Analysis**
One short paragraph.

**Environmental Impact**
One short paragraph.

**Cleanup Recommendation**
Two practical bullet points.

Mention that the detection system may not detect every object.
Keep the report under 150 words.
"""

    detected_list = ", ".join(
        f"{count}x {name}"
        for name, count in counts.items()
    )

    return f"""
You are an environmental monitoring assistant.

An AI vision model analyzed an underwater/coastal image and detected:

{detected_list}

Write a short environmental report with exactly three sections:

**Pollution Analysis**
Briefly explain what was detected.

**Environmental Impact**
Explain how these types of debris may affect marine life
or the marine ecosystem.

**Cleanup Recommendation**
Give 2 to 3 practical cleanup or monitoring actions.

Use simple language.
Do not invent additional detected objects.
Keep the report under 200 words.
"""


# ---------------------------------------------------------
# LOCAL FALLBACK REPORT
# ---------------------------------------------------------

def generate_fallback_report(counts):

    if not counts:

        return """
### Pollution Analysis

No marine debris was detected in the image.

### Environmental Impact

The analyzed area has no detectable debris based on the current
AI detection. However, the system may miss objects because of
image quality, object size, or visibility.

### Cleanup Recommendation

- Continue monitoring the area regularly.
- Perform periodic manual inspection and cleanup.
"""

    detected_items = []

    for name, count in counts.items():

        if count == 1:
            detected_items.append(f"1 {name}")
        else:
            detected_items.append(f"{count} {name}")

    items = ", ".join(detected_items)

    item_names = ", ".join(counts.keys())

    return f"""
### Pollution Analysis

The AI vision model detected the following marine debris:
**{items}**.

### Environmental Impact

Marine debris such as **{item_names}** can negatively affect
marine organisms through ingestion, entanglement, or habitat
pollution.

### Cleanup Recommendation

- Remove the detected debris from the affected area.
- Dispose of collected waste responsibly.
- Continue monitoring the area for additional marine debris.
"""


# ---------------------------------------------------------
# GENERATE REPORT
# ---------------------------------------------------------

def generate_report(counts, max_retries=1):

    api_key = os.environ.get("GEMINI_API_KEY")

    # If API key is missing, use local report
    if not api_key:

        print("Gemini API key not found.")
        print("Using local environmental report.")

        return generate_fallback_report(counts)

    try:

        print("Generating report with Gemini...")

        client = genai.Client(
            api_key=api_key
        )

        prompt = build_prompt(counts)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        # Make sure Gemini actually returned text
        if response.text:

            print("Gemini report generated successfully.")

            return response.text

        else:

            print("Gemini returned no text.")
            print("Using local environmental report.")

            return generate_fallback_report(counts)

    except Exception as e:

        print(f"Gemini unavailable: {e}")
        print("Using local environmental report instead.")

        return generate_fallback_report(counts)


# ---------------------------------------------------------
# COMMAND-LINE USAGE
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/marine_debris-2/weights/best.pt"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    args = parser.parse_args()

    # Run YOLO
    counts = run_detection(
        args.weights,
        args.image,
        args.conf
    )

    print(
        "\nDetections:",
        dict(counts) if counts else "none"
    )

    # Generate report
    print("\nGenerating environmental report...")

    report = generate_report(counts)

    print(
        "\n--- Environmental Report ---\n"
    )

    print(report)


# ---------------------------------------------------------
# START PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    main()