
try:
    import transformers
    print(f"Transformers imported successfully. Version: {transformers.__version__}")
    from transformers import CLIPModel, CLIPProcessor
    print("CLIPModel and CLIPProcessor imported successfully.")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
