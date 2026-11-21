from typer.testing import CliRunner

from not_trust_me_bro.cli import app

runner = CliRunner()


def test_config_masks_secrets(monkeypatch):
    monkeypatch.setenv("NTMB_OPENAI_API_KEY", "sk-test-not-real")
    from not_trust_me_bro.config import get_settings

    get_settings.cache_clear()
    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "llm_model" in result.output
    assert "sk-test-not-real" not in result.output
