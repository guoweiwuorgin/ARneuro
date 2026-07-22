try:
    from feature_extraction.task_info_extractor import (
        DeepSeekMethodTaskInfoExtractor,
        MethodTaskInfoExtractor,
        normalize_task_info,
    )
except Exception:
    from task_info_extractor import (
        DeepSeekMethodTaskInfoExtractor,
        MethodTaskInfoExtractor,
        normalize_task_info,
    )


def test_normalize_task_info_preserves_legacy_and_rich_fields():
    raw = {
        "tasks": [
            {
                "name": "Visual Word Recognition Task",
                "type": "Language sub-function test task",
                "clue\\_sentences": "Participants pressed a button for words.",
                "Task property on Production or Comprehension": "Comprehension",
                "Task property on attention engagement": "Active",
                "Task property on Linguistic Domain": ["Orthographic", "Lexical"],
                "Task property on Processing Type": ["Recognition/Identification"],
                "Task property on Stimulus modality": ["visual"],
                "stimulus_properties": {
                    "modalities": ["visual"],
                    "units": ["word"],
                    "types": ["text"],
                    "lexical_status": ["real words"],
                    "languages": ["English"],
                    "presentation": "screen",
                },
                "contrasts": [
                    {
                        "contrast_name": "words vs symbols",
                        "conditions_compared": ["words", "symbols"],
                        "direction": "words > symbols",
                        "exact_text": "words versus symbols",
                    }
                ],
            }
        ],
        "Language use": ["English"],
    }

    normalized = normalize_task_info("123", raw)

    assert normalized["PMID"] == "123"
    assert normalized["task_count"] == 1
    assert normalized["tasks"][0]["clue_sentences"].startswith("Participants")
    assert normalized["tasks"][0]["Task_property_on_Linguistic_Domain"] == [
        "Orthographic",
        "Lexical",
    ]
    assert normalized["Language_use"] == ["English"]
    assert normalized["tasks"][0]["contrasts"][0]["conditions_compared"] == [
        "words",
        "symbols",
    ]
    assert "downstream_feature_hints" in normalized["tasks"][0]


def test_control_flag_forces_control_task_type_and_boolean_normalization():
    normalized = normalize_task_info(
        "123",
        {
            "tasks": [
                {
                    "name": "Fixation Baseline",
                    "type": "Non-language cognitive task",
                    "is_control_task": True,
                    "experimental_design": {"scanner_execution": "true"},
                }
            ]
        },
    )

    task = normalized["tasks"][0]
    assert task["type"] == "Control experiment task"
    assert task["experimental_design"]["scanner_execution"] == "True"


def test_prompt_contains_required_naming_rules_and_complete_methods():
    extractor = MethodTaskInfoExtractor(config={})
    messages = extractor.build_messages(
        "123",
        "Participants generated as many words as possible in one minute.",
        "A study",
    )

    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "Verbal Fluency Task" in system
    assert "modality in name only when explicitly reported" in system
    assert "Complete_Methods_section" in user
    assert "as many words as possible" in user


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, model_name, responses, calls):
        self.model_name = model_name
        self.responses = responses
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return _FakeResponse(response)


class _FakeChat:
    def __init__(self, model_name, responses, calls):
        self.completions = _FakeCompletions(model_name, responses, calls)


class _FakeClient:
    def __init__(self, model_name, responses, calls):
        self.chat = _FakeChat(model_name, responses, calls)


def test_deepseek_retries_flash_once_then_uses_non_thinking_pro():
    extractor = DeepSeekMethodTaskInfoExtractor(
        api_key="test",
        max_retries=1,
        fallback_max_retries=0,
        request_interval_seconds=0,
    )
    calls = []
    clients = {
        "deepseek-v4-flash": _FakeClient(
            "deepseek-v4-flash",
            ["not json", "also not json"],
            calls,
        ),
        "deepseek-v4-pro": _FakeClient(
            "deepseek-v4-pro",
            ['{"tasks": [], "Language_use": []}'],
            calls,
        ),
    }
    extractor.get_client = lambda model_name=None: (
        clients[model_name or extractor.model_name],
        model_name or extractor.model_name,
    )

    raw, metadata = extractor.call_model(
        [{"role": "user", "content": "Methods"}]
    )

    assert raw["tasks"] == []
    assert [call["model"] for call in calls] == [
        "deepseek-v4-flash",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]
    assert all(
        call["extra_body"] == {"thinking": {"type": "disabled"}}
        for call in calls
    )
    assert metadata["fallback_used"] is True
    assert metadata["model"] == "deepseek-v4-pro"
    assert metadata["total_attempts"] == 3
    assert metadata["context_window_tokens"] == 384000
