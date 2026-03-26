from rag.generation.prompts import RAG_SYSTEM_PROMPT, get_rag_prompt


def test_prompt_template_has_required_variables():
    prompt = get_rag_prompt()
    input_vars = prompt.input_variables
    assert "context" in input_vars
    assert "question" in input_vars


def test_prompt_template_renders():
    prompt = get_rag_prompt()
    messages = prompt.format_messages(
        context="Nike was founded in 1964.", question="When was Nike founded?"
    )
    assert len(messages) == 2
    assert "Nike was founded in 1964" in messages[1].content
    assert "When was Nike founded?" in messages[1].content


def test_system_prompt_contains_grounding_instructions():
    assert "Only use information from the provided context" in RAG_SYSTEM_PROMPT
    assert "don't have enough information" in RAG_SYSTEM_PROMPT.lower()
