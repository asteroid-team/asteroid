from asteroid.scripts import asteroid_versions, asteroid_cli


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


def test_infer_device(monkeypatch):
    """Test that inference is performed on the PyTorch device given by '--device'.

    We can't properly test this in environments with only CPU device available.
    As an approximation we test that the '.to()' method of the model is called
    with the device given by '--device'.
    """
    # We can't use a real model to test this because calling .to() with a fake device
    # on a real model will fail.
    class FakeModel:
        def to(self, device):
            self.device = device

    fake_model = FakeModel()

    # Monkeypatch 'from_pretrained' to load our fake model.
    from asteroid.models import BaseModel

    monkeypatch.setattr(BaseModel, "from_pretrained", lambda *args, **kwargs: fake_model)

    # Note that this will issue a warning about the missing file.
    asteroid_cli.infer(
        ["--device", "cuda:42", "somemodel", "--files", "file_that_does_not_exist.wav"]
    )

    assert fake_model.device == "cuda:42"
