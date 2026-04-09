"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Running tests: pytest -xvs tests/llm_client/test_openai_generic_client.py

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message


def _make_client(
    content: str, prompt_tokens: int = 10, completion_tokens: int = 5
) -> OpenAIGenericClient:
    """Return an OpenAIGenericClient whose underlying API call returns *content*.

    The mock response includes a usage object with token counts so that
    the tuple return (dict, input_tokens, output_tokens) is fully exercised.
    """
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason='stop',
    )
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    mock_response = SimpleNamespace(choices=[choice], usage=usage)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch('openai.AsyncOpenAI', return_value=mock_openai):
        client = OpenAIGenericClient(config=LLMConfig(api_key='test'))
        client.client = mock_openai
        return client


MESSAGES = [Message(role='user', content='hello')]


class TestCodeFenceStripping:
    """OpenAIGenericClient strips markdown code fences before JSON parsing.

    On the devmao fork, _generate_response returns tuple[dict, int, int]
    (parsed JSON, input_tokens, output_tokens).
    """

    @pytest.mark.asyncio
    async def test_plain_json_unchanged(self):
        client = _make_client('{"key": "value"}')
        result, inp, out = await client._generate_response(MESSAGES)
        assert result == {'key': 'value'}
        assert inp == 10
        assert out == 5

    @pytest.mark.asyncio
    async def test_json_code_fence(self):
        client = _make_client('```json\n{"key": "value"}\n```')
        result, _, _ = await client._generate_response(MESSAGES)
        assert result == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_plain_code_fence(self):
        client = _make_client('```\n{"key": "value"}\n```')
        result, _, _ = await client._generate_response(MESSAGES)
        assert result == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_code_fence_with_extra_whitespace(self):
        client = _make_client('```json\n  {"key": "value"}  \n```')
        result, _, _ = await client._generate_response(MESSAGES)
        assert result == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_token_counts_returned(self):
        client = _make_client('{"ok": true}', prompt_tokens=42, completion_tokens=7)
        result, inp, out = await client._generate_response(MESSAGES)
        assert result == {'ok': True}
        assert inp == 42
        assert out == 7

    @pytest.mark.asyncio
    async def test_arbitrary_language_tag_fence(self):
        """Fences with non-json language tags (e.g. ```jsonl, ```text) are stripped."""
        client = _make_client('```jsonl\n{"key": "value"}\n```')
        result, _, _ = await client._generate_response(MESSAGES)
        assert result == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_mixed_case_language_tag(self):
        """Fences with mixed-case language tags (e.g. ```Json) are stripped."""
        client = _make_client('```Json\n{"key": "value"}\n```')
        result, _, _ = await client._generate_response(MESSAGES)
        assert result == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_empty_response_raises(self):
        """An empty LLM response raises ValueError (not JSONDecodeError)."""
        client = _make_client('')
        with pytest.raises(ValueError, match='empty response'):
            await client._generate_response(MESSAGES)

    @pytest.mark.asyncio
    async def test_whitespace_only_response_raises(self):
        """A whitespace-only LLM response raises ValueError."""
        client = _make_client('   \n  ')
        with pytest.raises(ValueError, match='empty response'):
            await client._generate_response(MESSAGES)
