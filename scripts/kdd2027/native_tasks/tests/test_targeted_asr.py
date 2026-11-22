import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1] / "tasr.py"
    spec = importlib.util.spec_from_file_location("targeted_asr_tests", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_primary_asr_is_target_hits_over_fixed_cohort():
    d = module().calculate(
        [
            {"clean": True, "adv": True},
            {"clean": False, "adv": True},
            {"clean": False, "adv": False},
        ],
        3,
    )
    assert d["targeted_success_numerator"] == 2
    assert d["targeted_success_denominator"] == 3
    assert d["ASR"] == d["TASR"] == 2 / 3
    assert d["new_target_hits"] == 1
    assert d["new_target_hit_rate_all"] == 1 / 3
    assert d["TASR_conditional_clean_not_target"] == 0.5


def test_missing_adversarial_judgment_is_not_failure_or_success():
    d = module().calculate([{"clean": False, "adv": True}, {"clean": False, "adv": None}], 2)
    assert d["ASR"] is None
    assert d["targeted_success_numerator"] == 1
    assert d["targeted_success_denominator"] == 2


def test_clean_eligibility_does_not_change_primary_targeted_denominator():
    d = module().calculate([{"clean": None, "adv": True}], 1)
    assert d["ASR"] == 1
    assert d["clean_target_hit_rate"] is None
    assert d["new_target_hit_rate_all"] is None


def test_untargeted_correctness_label_cannot_be_misread_as_target_hit():
    p = module().parse
    assert (
        p('{"target_hit":true,"reason":"Matches the specific motorcycle target scene"}')[
            "target_hit"
        ]
        is True
    )
    assert p('{"acceptable":false,"reason":"Wrong source content"}') is None
    assert p('{"target_hit":1,"reason":"not boolean"}') is None
