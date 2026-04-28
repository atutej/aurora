"""Tests for chat parsers and loss-mask generation."""

from types import SimpleNamespace

import torch

from aurora.data.parse import GeneralParser
from aurora.data.template import ChatTemplate


class CharTokenizer:
    """Minimal tokenizer with one token per character."""

    def __init__(self, chat_template: ChatTemplate, pad_token_id=None, unk_token_id=99):
        self.chat_template = chat_template
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token = None

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if tokenize:
            raise ValueError("test tokenizer only supports tokenize=False")

        parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                parts.append(f"<system>{content}{self.chat_template.end_of_turn_token}")
            elif role == "user":
                parts.append(
                    f"{self.chat_template.user_header}{content}"
                    f"{self.chat_template.end_of_turn_token}"
                )
            elif role == "assistant":
                parts.append(
                    f"{self.chat_template.assistant_header}{content}"
                    f"{self.chat_template.end_of_turn_token}"
                )
            else:
                raise ValueError(f"unexpected role: {role}")

        if kwargs.get("add_generation_prompt", False):
            parts.append(self.chat_template.assistant_header)

        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def __call__(
        self,
        text,
        max_length,
        truncation,
        return_tensors,
        add_special_tokens,
        return_offsets_mapping=False,
    ):
        if truncation:
            text = text[:max_length]

        input_ids = torch.arange(len(text), dtype=torch.long)[None, :]
        encoded = SimpleNamespace(input_ids=input_ids)

        if return_offsets_mapping:
            offsets = torch.tensor([(idx, idx + 1) for idx in range(len(text))], dtype=torch.long)
            encoded.offset_mapping = offsets[None, :]

        return encoded


def make_template() -> ChatTemplate:
    return ChatTemplate(
        assistant_header="<assistant>",
        user_header="<user>",
        system_prompt=None,
        end_of_turn_token="<eot>",
    )


def expected_mask_for_spans(text: str, spans: list[tuple[int, int]]) -> torch.Tensor:
    mask = torch.zeros(len(text), dtype=torch.long)
    for start, end in spans:
        mask[start:end] = 1
    return mask


class TestParserPadToken:
    def test_preserves_zero_pad_token_id(self):
        template = make_template()
        tokenizer = CharTokenizer(template, pad_token_id=0, unk_token_id=99)
        parser = GeneralParser(tokenizer, template)

        parser.parse("<user>hi<eot><assistant>ok<eot>", max_length=1024, preformatted=True)

        assert tokenizer.pad_token_id == 0

    def test_uses_unk_token_when_pad_token_is_missing(self):
        template = make_template()
        tokenizer = CharTokenizer(template, pad_token_id=None, unk_token_id=99)
        parser = GeneralParser(tokenizer, template)

        parser.parse("<user>hi<eot><assistant>ok<eot>", max_length=1024, preformatted=True)

        assert tokenizer.pad_token_id == 99


class TestGeneralParser:
    def test_parse_masks_all_assistant_turns(self):
        template = make_template()
        parser = GeneralParser(CharTokenizer(template), template)
        text = "<user>hi<eot><assistant>hello<eot><user>again<eot><assistant>bye<eot>"

        input_ids, loss_mask = parser.parse(text, max_length=1024, preformatted=True)

        first_start = text.index("hello")
        first_end = first_start + len("hello<eot>")
        second_start = text.index("bye")
        second_end = second_start + len("bye<eot>")
        assert torch.equal(input_ids, torch.arange(len(text), dtype=torch.long))
        assert torch.equal(
            loss_mask,
            expected_mask_for_spans(text, [(first_start, first_end), (second_start, second_end)]),
        )

    def test_parse_can_mask_last_assistant_turn_only(self):
        template = make_template()
        parser = GeneralParser(CharTokenizer(template), template)
        text = "<user>hi<eot><assistant>hello<eot><user>again<eot><assistant>bye<eot>"

        _, loss_mask = parser.parse(
            text,
            max_length=1024,
            preformatted=True,
            last_turn_only=True,
        )

        second_start = text.index("bye")
        second_end = second_start + len("bye<eot>")
        assert torch.equal(loss_mask, expected_mask_for_spans(text, [(second_start, second_end)]))

    def test_parse_clips_loss_mask_to_truncated_input(self):
        template = make_template()
        parser = GeneralParser(CharTokenizer(template), template)
        text = "<user>hi<eot><assistant>abcdef<eot>"
        max_length = text.index("def")

        input_ids, loss_mask = parser.parse(text, max_length=max_length, preformatted=True)

        assistant_start = text.index("abcdef")
        assert len(input_ids) == max_length
        assert torch.equal(
            loss_mask,
            expected_mask_for_spans(text[:max_length], [(assistant_start, max_length)]),
        )
