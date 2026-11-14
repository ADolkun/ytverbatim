import dataclasses
from typing import get_args

import pytest

import ytverbatim
from ytverbatim.text import numbers


def test_quantity_tokens_is_part_of_the_library_surface() -> None:
    assert {"quantity_tokens", "QToken", "QKind"} <= set(ytverbatim.__all__)
    assert ytverbatim.quantity_tokens is numbers.quantity_tokens
    assert ytverbatim.QToken is numbers.QToken
    assert ytverbatim.QKind is numbers.QKind


def test_the_token_shape_is_pinned() -> None:
    assert [(f.name, f.type) for f in dataclasses.fields(ytverbatim.QToken)] == [
        ("char_start", int),
        ("char_end", int),
        ("surface", str),
        ("value_itn", str),
        ("kind", numbers.QKind),
    ]
    assert get_args(ytverbatim.QKind) == ("number", "currency", "percent", "ticker", "cjk_quantity")


def test_tokens_carry_offsets_surface_value_and_kind() -> None:
    text = "SPY 涨了 3.5%，一百五十期，$1.5B"
    assert ytverbatim.quantity_tokens(text) == [
        ytverbatim.QToken(0, 3, "SPY", "SPY", "ticker"),
        ytverbatim.QToken(7, 11, "3.5%", "3.5%", "percent"),
        ytverbatim.QToken(12, 17, "一百五十期", "150期", "cjk_quantity"),
        ytverbatim.QToken(18, 23, "$1.5B", "$1500000000", "currency"),
    ]


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("2023,2024", [(0, 4, "2023", "2023", "number"), (5, 9, "2024", "2024", "number")]),
        ("3.35SMH", [(0, 4, "3.35", "3.35", "number"), (4, 7, "SMH", "SMH", "ticker")]),
        ("twenty-five", [(0, 11, "twenty-five", "25", "number")]),
        ("10BN", [(0, 4, "10BN", "10000000000", "number")]),
    ],
)
def test_pinned_token_examples(text: str, tokens: list[tuple[int, int, str, str, str]]) -> None:
    assert ytverbatim.quantity_tokens(text) == [ytverbatim.QToken(*token) for token in tokens]
