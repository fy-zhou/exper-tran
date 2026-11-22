import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1] / "asr.py"
    spec = importlib.util.spec_from_file_location("semantic_asr_tests", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_clean_correct_to_incorrect_is_success():
    pairs = [
        {"clean": True, "adv": False},
        {"clean": True, "adv": True},
        {"clean": False, "adv": False},
        {"clean": False, "adv": True},
    ]
    d = module().calculate(pairs, 4)
    assert d["attack_success_numerator"] == 1
    assert d["clean_correct_denominator"] == 2
    assert d["ASR_clean_correct"] == 0.5
    assert d["ASR_all_images_or_questions"] == 0.25
    assert d["adversarial_error_rate_all"] == 0.5


def test_missing_invalid_or_zero_denominator_never_becomes_zero_asr():
    m = module()
    assert m.calculate([{"clean": True, "adv": None}], 1)["ASR_clean_correct"] is None
    assert m.calculate([{"clean": True, "adv": False}], 2)["ASR_clean_correct"] is None
    d = m.calculate([{"clean": False, "adv": False}], 1)
    assert d["ASR_clean_correct"] is None
    assert d["zero_clean_correct_denominator"]


def test_only_strict_binary_labels_parse():
    p = module().parse
    assert p('{"acceptable":false,"reason":"wrong central object"}')["acceptable"] is False
    for text in [
        '{"acceptable":0,"reason":"x"}',
        '{"acceptable":true}',
        '{"score_A":0,"score_B":9}',
        "not JSON",
    ]:
        assert p(text) is None


def test_two_independent_batch_rows_and_left_padding_restore(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch
    from PIL import Image

    import primary_ml_cka.prompts.chat_templates as templates

    m = module()
    paths = []
    for i, color in enumerate(["white", "black"]):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (224, 224), color).save(p)
        paths.append(p)
    monkeypatch.setattr(templates, "render_chat_template", lambda *args, **kw: "rendered")
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    class Processor:
        tokenizer = SimpleNamespace(padding_side="right")

        def __call__(self, text, images, **kwargs):
            assert len(text) == len(images) == 2
            assert self.tokenizer.padding_side == "left"
            assert images[0].getpixel((0, 0)) == (255, 255, 255)
            assert images[1].getpixel((0, 0)) == (0, 0, 0)
            return {"input_ids": torch.tensor([[0, 4], [5, 6]])}

        def batch_decode(self, tokens, **kwargs):
            assert tokens.tolist() == [[10, 2], [11, 2]]
            return [
                '{"acceptable":true,"reason":"correct"}',
                '{"acceptable":false,"reason":"wrong object"}',
            ]

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 64
            return torch.cat([kwargs["input_ids"], torch.tensor([[10, 2], [11, 2]])], 1)

    p = Processor()
    packets = [{"image": str(x), "image_sha256": m.sha(x), "packet": {}} for x in paths]
    results = m.generate_batch(Model(), p, packets)
    assert m.parse(results[0]["text"])["acceptable"] is True
    assert m.parse(results[1]["text"])["acceptable"] is False
    assert p.tokenizer.padding_side == "right"
