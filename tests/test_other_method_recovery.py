from text_processing.other_method_recovery import (
    OtherMethodSectionRecoverer,
    split_other_into_blocks,
)


def test_recovers_explicit_method_headings_and_preserves_results():
    content = {
        "Methods": "### Subjects\nTen participants were enrolled.",
        "Other": (
            "### PET data acquisition\nScans were acquired on a Siemens scanner.\n\n"
            "### Data analysis\nImages were normalized and analysed with SPM.\n\n"
            "### Results overview\nActivation was significantly greater in condition A."
        ),
        "Methods_Hierarchy": [],
    }

    result = OtherMethodSectionRecoverer().recover(content)

    assert result.changed is True
    assert "PET data acquisition" in result.content["Methods"]
    assert "Data analysis" in result.content["Methods"]
    assert "Results overview" not in result.content["Methods"]
    assert "Results overview" in result.content["Other"]
    assert result.metadata["other_method_recovery"]["recovered_block_count"] == 2


def test_recovers_unheaded_prose_only_with_strong_evidence():
    content = {
        "Methods": "",
        "Other": (
            "Twenty healthy participants were recruited and gave written informed "
            "consent. Participants completed a lexical decision task during scanning.\n\n"
            "These findings suggest that language processing is broadly distributed."
        ),
    }

    result = OtherMethodSectionRecoverer().recover(content)

    assert "Twenty healthy participants" in result.content["Methods"]
    assert "These findings suggest" in result.content["Other"]


def test_long_unheaded_other_is_structurally_split_without_loss():
    method_paragraph = (
        "Participants were recruited and completed an experimental task. " * 80
    )
    discussion = "These findings suggest a theoretical explanation. " * 80
    blocks = split_other_into_blocks(
        f"{method_paragraph}\n\n{discussion}",
        max_unheaded_chars=900,
    )

    assert len(blocks) > 2
    assert all(len(block.content) <= 1000 for block in blocks)


def test_recovery_is_idempotent():
    content = {
        "Methods": "",
        "Other": "### Subjects\nTwelve healthy volunteers completed the task.",
        "Methods_Hierarchy": [],
    }
    recoverer = OtherMethodSectionRecoverer()
    first = recoverer.recover(content)
    second = recoverer.recover(first.content, first.metadata)

    assert first.changed is True
    assert second.changed is False
    assert second.content["Methods"] == first.content["Methods"]


def test_imaging_paper_title_and_supplementary_material_are_not_recovered():
    content = {
        "Methods": "",
        "Other": (
            "### The voice of emotion: an fMRI study of vocal expressions\n"
            "Author names and affiliations.\n\n"
            "### Supplementary Material\nAdditional figures and references.\n\n"
            "### fMRI data acquisition\nImages were acquired on a 3 T scanner."
        ),
    }

    result = OtherMethodSectionRecoverer().recover(content)

    assert "fMRI data acquisition" in result.content["Methods"]
    assert "The voice of emotion" in result.content["Other"]
    assert "Supplementary Material" in result.content["Other"]


def test_recovers_compound_participant_and_imaging_headings():
    content = {
        "Methods": "### Experimental procedure\nParticipants viewed words.",
        "Other": (
            "### Participants and procedure\n"
            "The data of 14 right-handed healthy participants were used.\n\n"
            "### fMRI acquisition and analysis\n"
            "Images were acquired on a 3T Siemens scanner and preprocessed."
        ),
    }

    result = OtherMethodSectionRecoverer().recover(content)

    assert result.changed is True
    assert "Participants and procedure" in result.content["Methods"]
    assert "14 right-handed" in result.content["Methods"]
    assert "fMRI acquisition and analysis" in result.content["Methods"]
    assert result.content["Other"] == ""


def test_preserves_previous_recovery_audit_in_history():
    recoverer = OtherMethodSectionRecoverer()
    first = recoverer.recover(
        {"Methods": "", "Other": "### Subjects\nTen participants were enrolled."}
    )
    second_content = dict(first.content)
    second_content["Other"] = (
        "### Participants and procedure\n"
        "Nine healthy participants completed the task."
    )
    second = recoverer.recover(second_content, first.metadata)

    assert second.changed is True
    assert len(second.metadata["other_method_recovery_history"]) == 1
    assert second.metadata["other_method_recovery"]["version"] == 2


def test_recovers_common_method_heading_variants():
    content = {
        "Methods": "",
        "Other": (
            "### General Procedure\nParticipants completed two runs.\n\n"
            "### Stimuli (see Table I for examples)\nForty words were presented.\n\n"
            "### MRI practice session\nParticipants practiced before scanning.\n\n"
            "### Scalp ERP Statistics\nERP amplitudes were analysed with ANOVA.\n\n"
            "### FUNCTIONAL LOCALIZER TASKS\nParticipants completed a language task.\n\n"
            "### Task performance\nAccuracy was significantly higher in condition A."
        ),
    }

    result = OtherMethodSectionRecoverer().recover(content)

    for heading in [
        "General Procedure",
        "Stimuli (see Table I for examples)",
        "MRI practice session",
        "Scalp ERP Statistics",
        "FUNCTIONAL LOCALIZER TASKS",
    ]:
        assert heading in result.content["Methods"]
    assert "Task performance" in result.content["Other"]


def test_uses_strong_prose_for_ambiguous_heading():
    content = {
        "Methods": "",
        "Other": (
            "### Session details\nTwenty healthy participants were recruited. "
            "Participants completed a lexical decision task during fMRI scanning."
        ),
    }

    result = OtherMethodSectionRecoverer().recover(content)

    assert "Session details" in result.content["Methods"]


def test_all_headings_ending_in_task_are_methods_but_results_remain_other():
    content = {
        "Methods": "",
        "Other": (
            "### Visual meaning task\nParticipants judged word associations.\n\n"
            "### Visual rhyming task\nParticipants judged whether words rhymed.\n\n"
            "### Auditory meaning and rhyming tasks\nWords were presented auditorily.\n\n"
            "### Task performance\nAccuracy was significantly above chance.\n\n"
            "### Results of the visual task\nActivation was greater for meaning.\n\n"
            "### Main network involved in the task\nActivation was widespread.\n\n"
            "### Brain Areas Activated in the Spoken Language Task\n"
            "Several regions were significant.\n\n"
            "### Neural correlates revealed by fMRI using a semantic task\n"
            "Author names and affiliations."
        ),
    }

    result = OtherMethodSectionRecoverer().recover(content)

    assert "Visual meaning task" in result.content["Methods"]
    assert "Visual rhyming task" in result.content["Methods"]
    assert "Auditory meaning and rhyming tasks" in result.content["Methods"]
    assert "Task performance" in result.content["Other"]
    assert "Results of the visual task" in result.content["Other"]
    assert "Main network involved in the task" in result.content["Other"]
    assert "Brain Areas Activated" in result.content["Other"]
    assert "Neural correlates revealed by fMRI" in result.content["Other"]
