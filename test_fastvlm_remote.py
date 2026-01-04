import logging
import sys
from orion.managers.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Testing FastVLM loading on remote...")
    
    try:
        manager = ModelManager.get_instance()
        logger.info(f"Device detected: {manager.device}")
        
        fastvlm = manager.fastvlm
        logger.info(f"FastVLM loaded: {fastvlm}")
        logger.info(f"FastVLM device: {fastvlm.device}")
        
        if hasattr(fastvlm, 'model'):
            logger.info(f"Model type: {type(fastvlm.model)}")
        
        logger.info("SUCCESS: FastVLM initialized correctly.")
        
    except Exception as e:
        logger.error(f"FAILURE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
