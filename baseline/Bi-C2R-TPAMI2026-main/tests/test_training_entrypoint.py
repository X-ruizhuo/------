from pathlib import Path


def test_training_entrypoint_defaults_to_base_config():
    train_script = Path(__file__).resolve().parents[1] / "continual_train.py"
    source = train_script.read_text(encoding="utf-8")

    assert "default='config/base.yml'" in source


def test_training_entrypoint_logs_feature_enhancer_scale():
    train_script = Path(__file__).resolve().parents[1] / "continual_train.py"
    source = train_script.read_text(encoding="utf-8")

    assert "def log_feature_enhancer_state" in source
    assert "feature_enhancer.res_scale raw=" in source


def test_resnet_uses_feature_enhancer_before_pooling():
    model_source = (
        Path(__file__).resolve().parents[1] / "reid" / "models" / "resnet.py"
    ).read_text(encoding="utf-8")

    assert "from .feature_enhancers import build_feature_enhancer" in model_source
    assert "self.feature_enhancer = build_feature_enhancer(cfg, channels=self.in_planes)" in model_source
    assert "x = self.feature_enhancer(x)" in model_source
    assert model_source.index("x = self.feature_enhancer(x)") < model_source.index(
        "global_feat = self.pooling_layer(x)"
    )


def test_run_scripts_select_baseline_and_eca_configs():
    root = Path(__file__).resolve().parents[1]

    assert "--logs-dir logs-res-setting1/ --setting 1" in (root / "run1.sh").read_text(
        encoding="utf-8"
    )
    assert "--logs-dir logs-res-setting2/ --setting 2" in (root / "run2.sh").read_text(
        encoding="utf-8"
    )
    assert "--config_file config/eca.yml" in (root / "run_eca1.sh").read_text(
        encoding="utf-8"
    )
    assert "--config_file config/eca.yml" in (root / "run_eca2.sh").read_text(
        encoding="utf-8"
    )


def test_eca_ablation_configs_and_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    expected_configs = [
        "eca_noop.yml",
        "eca_fixed_002.yml",
        "eca_fixed_005.yml",
        "eca_k5.yml",
        "eca_k7.yml",
        "eca_max_005.yml",
    ]

    for config_name in expected_configs:
        config_text = (root / "config" / config_name).read_text(encoding="utf-8")
        assert 'NAME: "eca_safe"' in config_text

    assert "CONFIG_NAME=${1:-eca_noop}" in (root / "run_eca_ablation1.sh").read_text(
        encoding="utf-8"
    )
    assert "--setting 1" in (root / "run_eca_ablation1.sh").read_text(
        encoding="utf-8"
    )
    assert "--setting 2" in (root / "run_eca_ablation2.sh").read_text(
        encoding="utf-8"
    )
