import subprocess
from asteroid.scripts import asteroid_versions


def test_asteroid_versions():
    versions = asteroid_versions.asteroid_versions()
    assert "Asteroid" in versions
    assert "PyTorch" in versions
    assert "PyTorch-Lightning" in versions


def test_print_versions():
    asteroid_versions.print_versions()


def test_asteroid_versions_without_git(monkeypatch):
    monkeypatch.setenv("PATH", "")
    asteroid_versions.asteroid_versions()


def cmd(cmd_str):
    str_list = cmd_str.split(" ")
    return subprocess.check_output(str_list).strip().decode("ascii", "ignore")


def test_register_sr():
    cmd("asteroid-register-sr --help")


def test_infer():
    cmd("asteroid-infer ")


def test_upload():
    cmd("asteroid-upload --help")
