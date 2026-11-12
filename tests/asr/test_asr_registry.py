import re

import pytest
from fake_engine import FakeEngine, definition

from ytverbatim.asr import registry
from ytverbatim.asr.protocol import EngineDef
from ytverbatim.errors import ConfigError, DependencyError

EXPECTED = {"w-prod", "w-vad", "q-base", "q-ctx", "f-aed", "m-4bit"}


def test_every_hypothesis_is_registered_with_a_pinned_revision() -> None:
    assert set(registry.HYPOTHESES) == EXPECTED
    for hypothesis_id, entry in registry.HYPOTHESES.items():
        assert entry.spec.id == hypothesis_id
        assert re.fullmatch(r"[0-9a-f]{40}", entry.spec.revision)
        assert entry.factory.startswith("ytverbatim.asr.engines.")


def test_whisper_control_decodes_whole_files_and_the_rest_share_the_chunk_plan() -> None:
    assert registry.get("w-prod").spec.max_chunk_s is None
    for hypothesis_id in EXPECTED - {"w-prod"}:
        assert registry.get(hypothesis_id).spec.max_chunk_s == 30.0


def test_variants_distinguish_what_the_spec_shares() -> None:
    assert registry.get("q-base").spec.repo == registry.get("q-ctx").spec.repo
    assert [registry.get(i).spec.supports_context for i in ("q-base", "q-ctx")] == [False, True]
    assert registry.get("w-prod").variant["mode"] == "whole_file"
    assert registry.get("w-vad").variant["mode"] == "chunk"
    assert registry.get("m-4bit").spec.sample_rate == 24_000
    assert registry.get("m-4bit").variant["tokenizer_repo"] == "mlx-community/MiMo-Audio-Tokenizer"


def test_unknown_hypothesis_lists_the_known_ones() -> None:
    with pytest.raises(ConfigError, match="w-prod"):
        registry.get("nope")


def test_build_engine_imports_the_factory_by_dotted_path() -> None:
    engine = registry.build_engine(definition(marker=1))
    assert isinstance(engine, FakeEngine)
    assert engine.variant == {"marker": 1}


def test_a_missing_adapter_is_a_dependency_error() -> None:
    missing = EngineDef(definition().spec, "ytverbatim.asr.engines.not_there.create", {})
    with pytest.raises(DependencyError, match="not_there"):
        registry.build_engine(missing)
    wrong_name = EngineDef(definition().spec, "fake_engine.nope", {})
    with pytest.raises(DependencyError, match="nope"):
        registry.build_engine(wrong_name)
