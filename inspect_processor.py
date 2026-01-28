
import inspect
from transformers import AutoProcessor

def inspect_processor():
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    
    sig = inspect.signature(processor.post_process_grounded_object_detection)
    print(f"Signature: {sig}")
    
    print("\nDocstring:")
    print(processor.post_process_grounded_object_detection.__doc__)

if __name__ == "__main__":
    inspect_processor()
