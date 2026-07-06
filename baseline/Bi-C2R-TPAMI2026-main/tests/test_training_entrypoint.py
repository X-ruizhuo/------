from pathlib import Path


def test_training_entrypoint_defaults_to_base_config():
    train_script = Path(__file__).resolve().parents[1] / "continual_train.py"
    source = train_script.read_text(encoding="utf-8")

    assert "default='config/base.yml'" in source
