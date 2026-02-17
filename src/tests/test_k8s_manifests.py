from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
K8S_DIR = ROOT / "k8s"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_k8s_required_manifests_exist():
    assert (K8S_DIR / "configmap.yaml").exists()
    assert (K8S_DIR / "deployment.yaml").exists()
    assert (K8S_DIR / "service.yaml").exists()
    assert (K8S_DIR / "secret.example.yaml").exists()


def test_deployment_has_health_checks_and_ports():
    deployment = _load_yaml(K8S_DIR / "deployment.yaml")
    assert deployment["kind"] == "Deployment"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    port_names = {p["name"] for p in container["ports"]}
    assert {"http", "metrics"}.issubset(port_names)
    assert "readinessProbe" in container
    assert "livenessProbe" in container


def test_service_exposes_http_and_metrics():
    service = _load_yaml(K8S_DIR / "service.yaml")
    assert service["kind"] == "Service"
    ports = {p["name"] for p in service["spec"]["ports"]}
    assert {"http", "metrics"}.issubset(ports)


def test_configmap_contains_runtime_flags():
    config_map = _load_yaml(K8S_DIR / "configmap.yaml")
    assert config_map["kind"] == "ConfigMap"
    data = config_map["data"]
    assert data["METRICS_ENABLED"] == "1"
    assert data["VERIFIER_BACKEND"] in {"heuristic", "torch"}
