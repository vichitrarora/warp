from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
import warnings

class QueryGenerator:
    def __init__(self, model_path: str, device: Optional[str] = None):
        # Suppress unnecessary warnings during initialization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left"  # Prevent padding-related warnings
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device or "auto",
                torch_dtype=self._get_optimal_dtype(),
            )
        self.model.eval()

    def _get_optimal_dtype(self):
        """Automatically select the best dtype for the hardware"""
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def _prepare_inputs(self, prompt: str):
        """Handle all tokenization warnings explicitly"""
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False  # Disable for modern models
        ).to(self.model.device)

    def generate_response(self, prompt: str, **generation_kwargs):
        """Main generation with all warnings handled"""
        default_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": 1,
        }
        generation_kwargs = {**default_kwargs, **generation_kwargs}

        inputs = self._prepare_inputs(prompt)

        # Critical: This context manager handles most warnings
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = self.model.generate(**inputs, **generation_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_mongo_query(self, nl_query: str, db_id: str, schema_info: str, **kwargs):
        prompt = f"""Below is an instruction... [your exact prompt template]"""
        return self.generate_response(prompt, **kwargs)