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
