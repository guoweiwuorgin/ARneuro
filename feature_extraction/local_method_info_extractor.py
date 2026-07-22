"""
Local-model Method-section information extraction.

This module reuses the MethodSectionInfoExtractor schema/prompt but replaces
online API calls with a local HuggingFace causal language model such as Qwen.
It is intended for server-side stable inference when online APIs are rate-limited.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from .method_info_extractor import MethodSectionInfoExtractor, extract_json_object
except Exception:
    from method_info_extractor import MethodSectionInfoExtractor, extract_json_object


class LocalModelMethodSectionInfoExtractor(MethodSectionInfoExtractor):
    """
    Extract Method-section information using a local HuggingFace chat model.

    Parameters:
        model_path:
            Local model directory, e.g. /storage/work/wuguowei/Bigmodel/Qwen3.5-9B.
        device_map:
            Passed to transformers AutoModelForCausalLM.from_pretrained.
        torch_dtype:
            "auto", "float16", "bfloat16", or "float32".
    """

    def __init__(
        self,
        model_path: str | Path,
        max_method_chars: int = 60000,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.9,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        request_sleep_seconds: float = 0.0,
        max_retries: int = 0,
    ) -> None:
        super().__init__(
            config={},
            client_type="local",
            model_name=str(model_path),
            max_method_chars=max_method_chars,
            request_sleep_seconds=request_sleep_seconds,
            max_retries=max_retries,
        )
        self.model_path = str(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.device_map = device_map
        self.torch_dtype_name = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.tokenizer = None
        self.model = None

    def get_client(self):  # noqa: D401 - compatible with parent naming.
        """Load and return the local tokenizer/model pair."""
        if self.model is not None and self.tokenizer is not None:
            return (self.tokenizer, self.model), self.model_path

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(self.torch_dtype_name).lower(), "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        return (self.tokenizer, self.model), self.model_path

    def _messages_to_prompt(self, messages: list[Dict[str, str]]) -> str:
        tokenizer, _model = self.get_client()[0]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}"
            for message in messages
        ) + "\n\nASSISTANT:\n"

    def call_model(self, messages: list[Dict[str, str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        (tokenizer, model), model_name = self.get_client()
        last_error = ""

        for retry_idx in range(self.max_retries + 1):
            try:
                import torch

                prompt = self._messages_to_prompt(messages)
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                self.call_count += 1

                do_sample = self.temperature > 0
                generation_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": do_sample,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    generation_kwargs.update(
                        {
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                        }
                    )

                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )

                generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                time.sleep(self.request_sleep_seconds)
                return extract_json_object(generated_text), {
                    "model": model_name,
                    "client_type": "local",
                    "error": "",
                    "retry_count": retry_idx,
                    "raw_generation_preview": generated_text[:1000],
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if retry_idx < self.max_retries:
                    time.sleep(min(2**retry_idx, 8))

        return {}, {
            "model": model_name,
            "client_type": "local",
            "error": last_error,
            "retry_count": self.max_retries,
        }
