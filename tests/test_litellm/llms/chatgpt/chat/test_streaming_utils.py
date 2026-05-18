"""
Tests for ChatGPTToolCallNormalizer.

Verifies that non-spec-compliant tool_call chunks from the ChatGPT backend API
are normalized to match the OpenAI streaming spec:
- Correct index assignment for parallel tool calls
- Deduplication of "closing" chunks with repeated id/name
"""

import pytest

from litellm.llms.chatgpt.chat.streaming_utils import ChatGPTToolCallNormalizer
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)


def _make_chunk(tool_calls=None, content=None):
    """Helper to build a ModelResponseStream chunk with tool_calls on the delta."""
    delta = Delta(
        content=content,
        role="assistant",
        tool_calls=tool_calls,
    )
    choice = StreamingChoices(delta=delta, index=0)
    return ModelResponseStream(choices=[choice])


def _make_tc(index=0, id=None, name=None, arguments=None):
    """Helper to build a ChatCompletionDeltaToolCall."""
    func = Function(name=name, arguments=arguments)
    return ChatCompletionDeltaToolCall(
        index=index,
        id=id,
        function=func,
        type="function" if id else None,
    )


class TestChatGPTToolCallNormalizer:
    """Test that the normalizer fixes ChatGPT-style tool_call streaming issues."""

    def test_single_tool_call_index_preserved(self):
        """A single tool call should get index=0."""
        chunks = [
            _make_chunk(
                tool_calls=[_make_tc(index=0, id="call_1", name="get_weather")]
            ),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"loc')]),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='ation": "NYC"}')]),
        ]
        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        assert len(results) == 3
        assert results[0].choices[0].delta.tool_calls[0].index == 0
        assert results[0].choices[0].delta.tool_calls[0].id == "call_1"
        assert results[1].choices[0].delta.tool_calls[0].index == 0
        assert results[2].choices[0].delta.tool_calls[0].index == 0

    def test_parallel_tool_calls_get_correct_indices(self):
        """
        ChatGPT sends all tool_calls with index=0. The normalizer should assign
        sequential indices: 0 for the first, 1 for the second.
        """
        chunks = [
            # First tool call: intro chunk with id + name
            _make_chunk(
                tool_calls=[_make_tc(index=0, id="call_aaa", name="get_weather")]
            ),
            # First tool call: arguments streaming
            _make_chunk(
                tool_calls=[_make_tc(index=0, arguments='{"location": "NYC"}')]
            ),
            # First tool call: duplicate closing chunk (id repeated) — should be skipped
            _make_chunk(
                tool_calls=[_make_tc(index=0, id="call_aaa", name="get_weather")]
            ),
            # Second tool call: intro chunk with id + name (index=0 from ChatGPT)
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_bbb", name="get_time")]),
            # Second tool call: arguments streaming
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"tz": "EST"}')]),
            # Second tool call: duplicate closing chunk — should be skipped
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_bbb", name="get_time")]),
        ]

        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        # 2 duplicate chunks should be skipped → 4 results
        assert len(results) == 4

        # First tool call chunks should have index=0
        assert results[0].choices[0].delta.tool_calls[0].index == 0
        assert results[0].choices[0].delta.tool_calls[0].id == "call_aaa"
        assert results[1].choices[0].delta.tool_calls[0].index == 0

        # Second tool call chunks should have index=1
        assert results[2].choices[0].delta.tool_calls[0].index == 1
        assert results[2].choices[0].delta.tool_calls[0].id == "call_bbb"
        assert results[3].choices[0].delta.tool_calls[0].index == 1

    def test_non_tool_call_chunks_pass_through(self):
        """Chunks without tool_calls should pass through unchanged."""
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
        ]
        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        assert len(results) == 2
        assert results[0].choices[0].delta.content == "Hello"
        assert results[1].choices[0].delta.content == " world"

    def test_empty_choices_pass_through(self):
        """Chunks with empty choices should pass through."""
        chunk = ModelResponseStream(choices=[])
        normalizer = ChatGPTToolCallNormalizer(iter([chunk]))
        results = list(normalizer)

        assert len(results) == 1

    def test_three_parallel_tool_calls(self):
        """Three parallel tool calls should get indices 0, 1, 2."""
        chunks = [
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_1", name="fn_a")]),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"a":1}')]),
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_2", name="fn_b")]),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"b":2}')]),
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_3", name="fn_c")]),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"c":3}')]),
        ]

        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        assert len(results) == 6
        # First tool call
        assert results[0].choices[0].delta.tool_calls[0].index == 0
        assert results[1].choices[0].delta.tool_calls[0].index == 0
        # Second tool call
        assert results[2].choices[0].delta.tool_calls[0].index == 1
        assert results[3].choices[0].delta.tool_calls[0].index == 1
        # Third tool call
        assert results[4].choices[0].delta.tool_calls[0].index == 2
        assert results[5].choices[0].delta.tool_calls[0].index == 2

    def test_all_duplicates_skipped(self):
        """If a chunk contains only duplicate tool_calls, the entire chunk is skipped."""
        chunks = [
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_x", name="fn")]),
            # Duplicate — same id seen before
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_x", name="fn")]),
        ]

        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        assert len(results) == 1
        assert results[0].choices[0].delta.tool_calls[0].id == "call_x"

    @pytest.mark.asyncio
    async def test_async_iteration(self):
        """The normalizer should work with async iteration."""

        async def async_gen():
            chunks = [
                _make_chunk(tool_calls=[_make_tc(index=0, id="call_a", name="fn_a")]),
                _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"x":1}')]),
                _make_chunk(tool_calls=[_make_tc(index=0, id="call_b", name="fn_b")]),
                _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"y":2}')]),
            ]
            for c in chunks:
                yield c

        normalizer = ChatGPTToolCallNormalizer(async_gen())
        results = []
        async for chunk in normalizer:
            results.append(chunk)

        assert len(results) == 4
        assert results[0].choices[0].delta.tool_calls[0].index == 0
        assert results[2].choices[0].delta.tool_calls[0].index == 1

    def test_getattr_proxies_to_stream(self):
        """Attribute access should be proxied to the underlying stream."""

        class FakeStream:
            custom_attr = "test_value"

            def __iter__(self):
                return iter([])

            def __next__(self):
                raise StopIteration

        normalizer = ChatGPTToolCallNormalizer(FakeStream())
        assert normalizer.custom_attr == "test_value"


def _make_finish_chunk(finish_reason):
    """Build a terminal-only chunk: empty delta, populated finish_reason."""
    delta = Delta(content=None, role=None, tool_calls=None)
    choice = StreamingChoices(delta=delta, index=0, finish_reason=finish_reason)
    return ModelResponseStream(choices=[choice])


class TestChatGPTToolCallFinishReasonFlip:
    """
    Regression for #27144 (second bug).

    The ChatGPT backend API can terminate a tool-calling stream with
    finish_reason="stop", which causes downstream consumers (OpenAI SDK,
    `stream_chunk_builder`, etc.) to miss the tool_call dispatch signal.
    Once the normalizer has observed any tool_call in the stream, a trailing
    "stop" must be upgraded to "tool_calls" — any other terminator must be
    left alone so truncation / safety signals still reach the caller.
    """

    def test_finish_reason_flipped_after_tool_calls_seen(self):
        chunks = [
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_1", name="simple_add")]),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"a":5}')]),
            _make_finish_chunk("stop"),
        ]
        results = list(ChatGPTToolCallNormalizer(iter(chunks)))

        assert results[-1].choices[0].finish_reason == "tool_calls"

    def test_finish_reason_stop_preserved_when_no_tool_calls(self):
        """Text-only stream → no flip."""
        chunks = [
            _make_chunk(content="hello"),
            _make_finish_chunk("stop"),
        ]
        results = list(ChatGPTToolCallNormalizer(iter(chunks)))

        assert results[-1].choices[0].finish_reason == "stop"

    def test_finish_reason_length_preserved_after_tool_calls(self):
        """A length cap during tool calls must still surface as 'length', not 'tool_calls'."""
        chunks = [
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_1", name="simple_add")]),
            _make_finish_chunk("length"),
        ]
        results = list(ChatGPTToolCallNormalizer(iter(chunks)))

        assert results[-1].choices[0].finish_reason == "length"

    def test_finish_reason_content_filter_preserved_after_tool_calls(self):
        chunks = [
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_1", name="simple_add")]),
            _make_finish_chunk("content_filter"),
        ]
        results = list(ChatGPTToolCallNormalizer(iter(chunks)))

        assert results[-1].choices[0].finish_reason == "content_filter"

    def test_finish_reason_none_preserved(self):
        """Intermediate chunks with finish_reason=None must not get flipped."""
        chunks = [
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_1", name="simple_add")]),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"a":5}')]),
        ]
        results = list(ChatGPTToolCallNormalizer(iter(chunks)))

        # No flip on intermediate chunks
        for r in results:
            assert r.choices[0].finish_reason is None
