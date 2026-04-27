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

import json
import logging
import re
import typing
from typing import Any, ClassVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'

# Regex to strip markdown code fences (```json ... ```, ```JSON ... ```,
# ```jsonl ... ```, or bare ``` ... ```).  Case-insensitive and accepts any
# language tag so we catch all variants Claude might produce.
_MD_FENCE_RE = re.compile(
    r'^\s*```\w*\s*\n?(.*?)\n?\s*```\s*$',
    re.DOTALL,
)


def _extract_first_json_value(text: str) -> str | None:
    """Find the first complete top-level JSON object or array in `text`.

    Walks the string char-by-char tracking brace/bracket depth, with
    string-literal awareness (so braces inside `"..."` strings do not
    affect depth). Returns the substring of the first balanced
    {...} or [...], or None if no complete JSON value is found.

    Recovers responses where Claude emits valid JSON followed by
    reasoning prose like `{...}\\n\\nWait, let me reconsider...`,
    which Anthropic models do despite `response_format: json_object`
    when JSON-schema enforcement is downgraded (e.g. via the GitHub
    Copilot proxy path).
    """
    s = text.strip()
    if not s:
        return None
    start_chars = {'{': '}', '[': ']'}
    start = -1
    open_char = ''
    for i, c in enumerate(s):
        if c in start_chars:
            start = i
            open_char = c
            break
    if start == -1:
        return None
    close_char = start_chars[open_char]

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


class OpenAIGenericClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, client: typing.Any = None):
            Initializes the OpenAIClient with the provided configuration, cache setting, and client.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
    ):
        """
        Initialize the OpenAIGenericClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 16384 (16K) for better compatibility with local models.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Override max_tokens to support higher limits for local models
        self.max_tokens = max_tokens

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> tuple[dict[str, typing.Any], int, int]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        try:
            # Prepare response format
            response_format: dict[str, Any] = {'type': 'json_object'}
            if response_model is not None:
                schema_name = getattr(response_model, '__name__', 'structured_response')
                json_schema = response_model.model_json_schema()
                response_format = {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': schema_name,
                        'schema': json_schema,
                    },
                }

            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,  # type: ignore[arg-type]
            )

            # --- diagnostic logging ---
            raw_content = response.choices[0].message.content if response.choices else None
            finish_reason = response.choices[0].finish_reason if response.choices else None
            logger.debug(
                'LLM response: model=%s, content type=%s, len=%s, '
                'finish_reason=%s, usage=%s, first 500 chars: %.500r',
                self.model or DEFAULT_MODEL,
                type(raw_content).__name__,
                len(raw_content) if raw_content else 'None',
                finish_reason,
                response.usage,
                raw_content,
            )

            result = raw_content or ''

            # --- token tracking (Copilot/OpenRouter backends) ---
            input_tokens = 0
            output_tokens = 0
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0

            # Guard against empty/None responses before attempting to parse.
            if not result.strip():
                raise ValueError('LLM returned an empty response')

            # Strip markdown code fences that some models (e.g. Claude via
            # Copilot) wrap around JSON responses despite json_object format.
            fence_match = _MD_FENCE_RE.match(result)
            if fence_match:
                logger.debug(
                    'Stripped markdown code fence from LLM response (model=%s, len=%d)',
                    self.model or DEFAULT_MODEL,
                    len(result),
                )
                result = fence_match.group(1)

            parsed = json.loads(result)
            return parsed, input_tokens, output_tokens
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except json.JSONDecodeError as e:
            extracted = _extract_first_json_value(result)
            if extracted is not None:
                try:
                    parsed = json.loads(extracted)
                except json.JSONDecodeError:
                    extracted = None
                else:
                    logger.warning(
                        'Recovered JSON via prefix-extraction fallback '
                        '(model=%s, raw_len=%d, extracted_len=%d). '
                        'Indicates schema enforcement is downgraded.',
                        self.model or DEFAULT_MODEL,
                        len(result),
                        len(extracted),
                    )
                    return parsed, input_tokens, output_tokens
            logger.error(
                'LLM response is not valid JSON (model=%s, first 200 chars: %.200r): %s',
                self.model or DEFAULT_MODEL,
                result,
                e,
            )
            raise
        except Exception as e:
            logger.error(
                'Error processing LLM response (model=%s): %s',
                self.model or DEFAULT_MODEL,
                e,
            )
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'openai',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            retry_count = 0
            last_error = None

            while retry_count <= self.MAX_RETRIES:
                try:
                    response, input_tokens, output_tokens = await self._generate_response(
                        messages, response_model, max_tokens=max_tokens, model_size=model_size
                    )
                    self.token_tracker.record(prompt_name or 'unknown', input_tokens, output_tokens)
                    return response
                except (RateLimitError, RefusalError):
                    # These errors should not trigger retries
                    span.set_status('error', str(last_error))
                    raise
                except (
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ):
                    # Let OpenAI's client handle these retries
                    span.set_status('error', str(last_error))
                    raise
                except Exception as e:
                    last_error = e

                    # Don't retry if we've hit the max retries
                    if retry_count >= self.MAX_RETRIES:
                        logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                        span.set_status('error', str(e))
                        span.record_exception(e)
                        raise

                    retry_count += 1

                    # Classify the error for clearer logging
                    err_type = type(e).__name__
                    if isinstance(e, json.JSONDecodeError):
                        retry_reason = 'invalid JSON (likely markdown code fence wrapping)'
                    elif 'validation error' in str(e).lower():
                        retry_reason = f'schema validation mismatch — {str(e).splitlines()[0]}'
                    else:
                        retry_reason = str(e)

                    # Construct a detailed error message for the LLM
                    error_context = (
                        f'The previous response attempt was invalid. '
                        f'Error type: {err_type}. '
                        f'Error details: {str(e)}. '
                        f'Please try again with a valid response, ensuring the output matches '
                        f'the expected format and constraints.'
                    )

                    error_message = Message(role='user', content=error_context)
                    messages.append(error_message)
                    logger.warning(
                        'Retrying after %s (attempt %d/%d): %s',
                        retry_reason,
                        retry_count,
                        self.MAX_RETRIES,
                        err_type,
                    )

            # If we somehow get here, raise the last error
            span.set_status('error', str(last_error))
            raise last_error or Exception('Max retries exceeded with no specific error')
