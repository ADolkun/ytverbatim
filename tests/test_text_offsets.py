from ytverbatim.contract import Flag
from ytverbatim.text.offsets import ChangedRun, changed_runs, remap_flags


def flag_on(text: str, part: str, kind: str = "decoder_anomaly") -> Flag:
    start = text.index(part)
    return Flag.model_validate({"kind": kind, "char_start": start, "char_end": start + len(part), "note": part})


def spans(text: str, flags: tuple[Flag, ...]) -> list[str]:
    return [text[f.char_start : f.char_end] for f in flags]


def test_marks_inserted_on_both_sides_and_elsewhere_stay_outside_the_flag() -> None:
    before = "今天我们聊聊GAAP利润然后看估值"
    after = "今天，我们聊聊GAAP，利润。然后看估值。"
    moved = remap_flags([flag_on(before, "GAAP"), flag_on(before, "估值")], [(before, after)])
    assert spans(after, moved) == ["GAAP", "估值"]


def test_flags_follow_several_removals_in_one_text() -> None:
    before = "um the GAAP number um rose"
    after = "the GAAP number rose"
    assert spans(after, remap_flags([flag_on(before, "GAAP"), flag_on(before, "rose")], [(before, after)])) == [
        "GAAP",
        "rose",
    ]


def test_a_flag_over_a_replacement_stretches_and_a_deleted_flag_is_dropped() -> None:
    before = "涨了十五个百分点 uh 好"
    after = "涨了50个百分点 好"
    moved = remap_flags([flag_on(before, "五"), flag_on(before, "uh")], [(before, after)])
    assert spans(after, moved) == ["50"]


def test_changes_chain_in_order() -> None:
    first, second, third = "a GAP b", "a GAAP b", "a GAAP b."
    assert spans(third, remap_flags([flag_on(first, "GAP")], [(first, second), (second, third)])) == ["GAAP"]


def test_changed_runs_are_whole_words_and_cjk_characters() -> None:
    assert changed_runs("the GAP rose", "the GAAP rose") == [ChangedRun(4, 8, "GAP", "GAAP")]
    assert changed_runs("适应率很高", "市盈率很高") == [ChangedRun(0, 2, "适应", "市盈")]
    assert changed_runs("大家好 欢迎", "大家好，欢迎。") == [ChangedRun(3, 4, " ", "，"), ChangedRun(6, 7, "", "。")]
    assert changed_runs("same", "same") == []


def test_rare_cjk_characters_are_units_of_their_own() -> None:
    # U+3400 (Extension A), U+F900 (compatibility) and U+20000 (Extension B) next to Latin letters.
    assert changed_runs("a\u3400b", "a\u3400c") == [ChangedRun(2, 3, "b", "c")]
    assert changed_runs("x\uf900\U00020000", "x\uf900\U00020001") == [ChangedRun(2, 3, "\U00020000", "\U00020001")]
