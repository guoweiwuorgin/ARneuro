"""
Reusable serial API extraction for structured Methods information.

SerialAPIMethodSectionInfoExtractor contains provider-independent extraction,
retry, persistence, and resume behavior. GLMMethodSectionInfoExtractor is a
small BigModel configuration wrapper. The same base can later run DeepSeek.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from .method_info_extractor import (
        MethodInfoExtractionResult,
        MethodSectionInfoExtractor,
        extract_json_object,
        pmid_from_path,
        read_json,
        write_csv,
        write_json,
    )
except Exception:
    from method_info_extractor import (
        MethodInfoExtractionResult,
        MethodSectionInfoExtractor,
        extract_json_object,
        pmid_from_path,
        read_json,
        write_csv,
        write_json,
    )


class SerialAPIMethodSectionInfoExtractor(MethodSectionInfoExtractor):
    """
    Provider-independent, strictly serial API extractor.

    Example DeepSeek construction:

        SerialAPIMethodSectionInfoExtractor(
            config={"deepseek_api_key": "..."},
            client_type="deepseek",
            model_name="deepseek-chat",
        )
    """

    def __init__(
        self,
        config: Dict[str, Any],
        client_type: str,
        model_name: str,
        max_method_chars: int = 50000,
        request_interval_seconds: float = 0.0,
        max_retries: int = 5,
        max_tokens: int = 8192,
        short_methods_threshold: int = 3000,
        supplementary_section_chars: int = 12000,
    ) -> None:
        super().__init__(
            config=config,
            client_type=client_type,
            model_name=model_name,
            max_method_chars=max_method_chars,
            request_sleep_seconds=request_interval_seconds,
            max_retries=max_retries,
            short_methods_threshold=short_methods_threshold,
            supplementary_section_chars=supplementary_section_chars,
            max_tokens=max_tokens,
        )
        self._last_request_finished_at = 0.0

    def _wait_for_serial_slot(self) -> None:
        elapsed = time.monotonic() - self._last_request_finished_at
        remaining = self.request_sleep_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return (
            "ratelimit" in text
            or "rate limit" in text
            or "too many requests" in text
            or "error code: 429" in text
            or "'code': '429'" in text
        )

    def call_model(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        client, model_name = self.get_client()
        last_error = ""

        for retry_idx in range(self.max_retries + 1):
            try:
                self._wait_for_serial_slot()
                self.call_count += 1
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    stream=False,
                )
                self._last_request_finished_at = time.monotonic()
                raw_text = response.choices[0].message.content or ""
                return extract_json_object(raw_text), {
                    "model": model_name,
                    "client_type": self.client_type,
                    "error": "",
                    "retry_count": retry_idx,
                    "raw_generation_preview": raw_text[:1000],
                }
            except Exception as exc:
                self._last_request_finished_at = time.monotonic()
                last_error = f"{type(exc).__name__}: {exc}"
                if retry_idx >= self.max_retries:
                    break

                if self._is_rate_limit_error(exc):
                    delay = min(30 * (2**retry_idx), 300)
                else:
                    delay = min(5 * (2**retry_idx), 60)
                delay += random.uniform(0.0, 2.0)
                print(
                    f"  Request failed ({last_error}); retry "
                    f"{retry_idx + 1}/{self.max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)

        return {}, {
            "model": model_name,
            "client_type": self.client_type,
            "error": last_error,
            "retry_count": self.max_retries,
        }

    def coerce_raw_json(self, pmid: str, raw_json: Dict[str, Any]) -> Dict[str, Any]:
        data = super().coerce_raw_json(pmid, raw_json)
        case_values = [
            str(data.get("case_report", "")).strip(),
            str(data.get("Case report", "")).strip(),
        ]
        resolved_case = next(
            (value for value in case_values if value in {"True", "False"}),
            "unknown",
        )
        data["case_report"] = resolved_case
        data["Case report"] = resolved_case
        return data

    @staticmethod
    def _existing_result_is_success(existing: Dict[str, Any]) -> bool:
        metadata = existing.get("metadata", {})
        extracted = existing.get("extracted_json", {})
        return (
            isinstance(metadata, dict)
            and metadata.get("status") == "success"
            and isinstance(extracted, dict)
            and bool(extracted)
        )

    def extract_from_content_file(
        self,
        content_path: Path,
        output_json_dir: Optional[Path] = None,
        overwrite: bool = False,
    ) -> MethodInfoExtractionResult:
        pmid = pmid_from_path(content_path)
        output_json_path = None
        if output_json_dir is not None:
            output_json_path = output_json_dir / f"paper_{pmid}_method_info.json"
            if output_json_path.exists() and not overwrite:
                existing = read_json(output_json_path)
                if self._existing_result_is_success(existing):
                    return MethodInfoExtractionResult(
                        pmid=pmid,
                        raw_json=existing.get("extracted_json", {}),
                        flat_record=existing.get("flat_record", {}),
                        metadata=existing.get("metadata", {}),
                    )

        result = self.extract_from_content(
            pmid=pmid,
            content=read_json(content_path),
            source_file=str(content_path),
        )
        if output_json_path is not None:
            write_json(
                {
                    "pmid": result.pmid,
                    "extracted_json": result.raw_json,
                    "flat_record": result.flat_record,
                    "metadata": result.metadata,
                },
                output_json_path,
            )
        return result

    def extract_directory_serial(
        self,
        segmented_dir: Path,
        output_json_dir: Path,
        output_csv_path: Path,
        limit: Optional[int] = None,
        overwrite: bool = False,
        skip_pmids: Optional[Iterable[str]] = None,
        include_pmids: Optional[Iterable[str]] = None,
        csv_checkpoint_every: int = 1,
    ) -> List[Dict[str, Any]]:
        skip_set = {str(pmid) for pmid in (skip_pmids or [])}
        include_set = (
            {str(pmid) for pmid in include_pmids}
            if include_pmids is not None
            else None
        )
        content_files = [
            path
            for path in sorted(
                segmented_dir.glob("paper_*_structured_content.json")
            )
            if pmid_from_path(path) not in skip_set
            and (
                include_set is None
                or pmid_from_path(path) in include_set
            )
        ]
        if limit is not None:
            content_files = content_files[:limit]

        records: List[Dict[str, Any]] = []
        for index, content_path in enumerate(content_files, start=1):
            print(
                f"[{index}/{len(content_files)}] {self.model_name} serial extraction: "
                f"{content_path.name}"
            )
            result = self.extract_from_content_file(
                content_path=content_path,
                output_json_dir=output_json_dir,
                overwrite=overwrite,
            )
            records.append(result.flat_record)
            if (
                csv_checkpoint_every > 0
                and (
                    index % csv_checkpoint_every == 0
                    or index == len(content_files)
                )
            ):
                write_csv(records, output_csv_path)
            print(
                f"  PMID {result.pmid}: {result.metadata.get('status')} "
                f"(retry={result.metadata.get('retry_count', 0)}; "
                f"sections={result.metadata.get('source_sections_used', [])})"
            )
        return records


class GLMMethodSectionInfoExtractor(SerialAPIMethodSectionInfoExtractor):
    """BigModel GLM-4.5-Air configuration of the reusable serial extractor."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "GLM-4.5-Air",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config={
                "glm_api_key": api_key,
                "glm_model_name": model_name,
            },
            client_type="glm",
            model_name=model_name,
            **kwargs,
        )


class DeepSeekMethodSectionInfoExtractor(SerialAPIMethodSectionInfoExtractor):
    """DeepSeek-V4-Flash configuration with thinking explicitly disabled."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-v4-flash",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config={
                "deepseek_api_key": api_key,
            },
            client_type="deepseek",
            model_name=model_name,
            **kwargs,
        )

    def call_model(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        client, model_name = self.get_client()
        last_error = ""

        for retry_idx in range(self.max_retries + 1):
            try:
                self._wait_for_serial_slot()
                self.call_count += 1
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    stream=False,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                self._last_request_finished_at = time.monotonic()
                raw_text = response.choices[0].message.content or ""
                return extract_json_object(raw_text), {
                    "model": model_name,
                    "client_type": self.client_type,
                    "thinking": "disabled",
                    "error": "",
                    "retry_count": retry_idx,
                    "raw_generation_preview": raw_text[:1000],
                }
            except Exception as exc:
                self._last_request_finished_at = time.monotonic()
                last_error = f"{type(exc).__name__}: {exc}"
                if retry_idx >= self.max_retries:
                    break
                if self._is_rate_limit_error(exc):
                    delay = min(30 * (2**retry_idx), 300)
                else:
                    delay = min(5 * (2**retry_idx), 60)
                delay += random.uniform(0.0, 2.0)
                print(
                    f"  Request failed ({last_error}); retry "
                    f"{retry_idx + 1}/{self.max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)

        return {}, {
            "model": model_name,
            "client_type": self.client_type,
            "thinking": "disabled",
            "error": last_error,
            "retry_count": self.max_retries,
        }
