import os
import time
import json
import google.generativeai as genai
from pathlib import Path

# Configure API key
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    exit(1)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Paths
workspace_root = Path(__file__).resolve().parents[1]
video_path = workspace_root / "data/examples/test.mp4"
output_path = workspace_root / "results/test_run/gemini_analysis.json"

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")
    print()

def analyze_video():
    # Upload video
    print(f"Uploading {video_path}...")
    video_file = upload_to_gemini(video_path, mime_type="video/mp4")

    # Wait for processing
    wait_for_files_active([video_file])

    # Create the prompt
    prompt = """
    Analyze this video and identify the distinct physical entities (objects) present.
    For each entity, provide:
    1. "object_class": A short class name (e.g., "chair", "person", "potted plant").
    2. "description": A detailed visual description of the specific instance of the object in the video (color, material, location, distinguishing features).
    3. "approximate_time_seen": When it appears in the video (e.g., "start", "middle", "throughout").

    Output the result as a JSON object with a key "entities" containing a list of these objects.
    Example:
    {
        "entities": [
            {"object_class": "chair", "description": "A black office chair with mesh back...", "approximate_time_seen": "start"},
            ...
        ]
    }
    """

    # Generate content
    print("Generating analysis with Gemini 1.5 Pro...")
    
    # List models to debug
    print("Available models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

    model = genai.GenerativeModel(model_name="gemini-2.5-pro")
    
    response = model.generate_content(
        [video_file, prompt],
        generation_config={"response_mime_type": "application/json"}
    )

    # Save result
    try:
        result_json = json.loads(response.text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result_json, f, indent=2)
        print(f"Analysis saved to {output_path}")
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON response.")
        print("Raw response:", response.text)

    # Cleanup
    print("Deleting uploaded file...")
    genai.delete_file(video_file.name)

if __name__ == "__main__":
    analyze_video()
