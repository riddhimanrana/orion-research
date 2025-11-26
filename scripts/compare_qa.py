import os
import time
import json
import argparse
import google.generativeai as genai
from pathlib import Path
import sys

# Add workspace root to path
workspace_root = Path(__file__).resolve().parents[1]
sys.path.append(str(workspace_root))

from orion.query.query_engine import QueryEngine
from orion.query.index import VideoIndex

# Configure API key
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    exit(1)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

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

def get_gemini_answer(video_file, question):
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content([video_file, question])
    return response.text

def run_comparison():
    # Setup paths
    test_video_path = workspace_root / "data/examples/test.mp4"
    main_video_path = workspace_root / "data/examples/video.mp4"
    
    test_index_path = workspace_root / "results/test_run/video_index.db"
    main_index_path = workspace_root / "results/full_video_analysis/video_index.db"

    # Initialize local engines
    print("Initializing local Query Engines...")
    test_index = VideoIndex(test_index_path, test_video_path)
    test_index.connect()
    test_engine = QueryEngine(test_index, test_video_path)

    main_index = VideoIndex(main_index_path, main_video_path)
    main_index.connect()
    main_engine = QueryEngine(main_index, main_video_path)

    # Upload videos to Gemini
    print("Uploading videos to Gemini...")
    test_video_file = upload_to_gemini(test_video_path, mime_type="video/mp4")
    main_video_file = upload_to_gemini(main_video_path, mime_type="video/mp4")
    
    wait_for_files_active([test_video_file, main_video_file])

    questions = [
        # test.mp4 questions
        {"video": "test.mp4", "q": "Is there a bird in the video?", "engine": test_engine, "g_file": test_video_file},
        {"video": "test.mp4", "q": "Where is the cat?", "engine": test_engine, "g_file": test_video_file},
        {"video": "test.mp4", "q": "Is there a traffic light?", "engine": test_engine, "g_file": test_video_file},
        {"video": "test.mp4", "q": "Describe the potted plant.", "engine": test_engine, "g_file": test_video_file},
        {"video": "test.mp4", "q": "Is there a chandelier?", "engine": test_engine, "g_file": test_video_file},
        {"video": "test.mp4", "q": "Where is the computer monitor?", "engine": test_engine, "g_file": test_video_file},
        {"video": "test.mp4", "q": "Is there a keyboard?", "engine": test_engine, "g_file": test_video_file},
        {"video": "test.mp4", "q": "What color is the sofa?", "engine": test_engine, "g_file": test_video_file},
        {"video": "test.mp4", "q": "Is there a refrigerator?", "engine": test_engine, "g_file": test_video_file},
        {"video": "test.mp4", "q": "How many chairs are there?", "engine": test_engine, "g_file": test_video_file},
        
        # video.mp4 questions
        {"video": "video.mp4", "q": "Is there a truck?", "engine": main_engine, "g_file": main_video_file},
        {"video": "video.mp4", "q": "Where is the book?", "engine": main_engine, "g_file": main_video_file},
        {"video": "video.mp4", "q": "Describe the notebook.", "engine": main_engine, "g_file": main_video_file},
        {"video": "video.mp4", "q": "Is there a person?", "engine": main_engine, "g_file": main_video_file},
        {"video": "video.mp4", "q": "What is the person holding?", "engine": main_engine, "g_file": main_video_file},
        {"video": "video.mp4", "q": "Is there a bird?", "engine": main_engine, "g_file": main_video_file},
        {"video": "video.mp4", "q": "Is there a keyboard?", "engine": main_engine, "g_file": main_video_file},
        {"video": "video.mp4", "q": "What color is the backpack?", "engine": main_engine, "g_file": main_video_file},
        {"video": "video.mp4", "q": "Is there a bed?", "engine": main_engine, "g_file": main_video_file},
        {"video": "video.mp4", "q": "Is there a tv?", "engine": main_engine, "g_file": main_video_file},
    ]

    print(f"\n{'Video':<10} | {'Question':<30} | {'Orion Answer':<40} | {'Gemini Answer':<40}")
    print("-" * 130)

    results = []

    for item in questions:
        video = item["video"]
        q = item["q"]
        engine = item["engine"]
        g_file = item["g_file"]

        # Orion Answer
        try:
            orion_resp = engine.query(q)
            orion_ans = orion_resp.answer
        except Exception as e:
            orion_ans = f"Error: {e}"

        # Gemini Answer
        try:
            gemini_ans = get_gemini_answer(g_file, q)
            # Clean up newlines for table
            gemini_ans = gemini_ans.replace("\n", " ").strip()[:100] + "..."
        except Exception as e:
            gemini_ans = f"Error: {e}"

        # Truncate Orion answer for table
        orion_ans_display = orion_ans.replace("\n", " ").strip()[:100] + "..."

        print(f"{video:<10} | {q:<30} | {orion_ans_display:<40} | {gemini_ans:<40}")
        
        results.append({
            "video": video,
            "question": q,
            "orion_answer": orion_ans,
            "gemini_answer": gemini_ans
        })

    # Save detailed results
    with open("qa_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to qa_comparison_results.json")

if __name__ == "__main__":
    run_comparison()
