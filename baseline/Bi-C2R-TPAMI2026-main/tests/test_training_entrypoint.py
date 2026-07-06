from pathlib import Path


def test_training_entrypoint_defaults_to_base_config():
    train_script = Path(__file__).resolve().parents[1] / "continual_train.py"
    source = train_script.read_text(encoding="utf-8")

    assert "default='config/base.yml'" in source


def test_run_scripts_select_baseline_and_hpa_configs():
    root = Path(__file__).resolve().parents[1]

    assert "--config_file config/base.yml" in (root / "run1.sh").read_text(encoding="utf-8")
    assert "--config_file config/base.yml" in (root / "run2.sh").read_text(encoding="utf-8")
    assert "--config_file config/hpa.yml" in (root / "run_hpa1.sh").read_text(encoding="utf-8")
    assert "--config_file config/hpa.yml" in (root / "run_hpa2.sh").read_text(encoding="utf-8")
    assert "--config_file config/hpa_full.yml" in (root / "run_hpa_full1.sh").read_text(encoding="utf-8")
    assert "--config_file config/hpa_full.yml" in (root / "run_hpa_full2.sh").read_text(encoding="utf-8")
