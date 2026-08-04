import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("MedicalRAG")

logging.getLogger("httpx").disabled = True
logging.getLogger("httpcore").disabled = True
logging.getLogger("huggingface_hub").disabled = True
logging.getLogger("transformers").disabled = True