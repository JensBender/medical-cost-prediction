import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_inference_imports_do_not_load_training_modules():
    """Keep production inference imports independent of training-only modules.

    The app imports ``src.prediction`` and ``src.explainability`` while serving
    requests. If those imports load ``src.modeling``, MLflow, or DVC, the app
    could require packages intentionally limited to the training environment.
    Run the imports in a fresh Python process so modules loaded by earlier tests
    cannot affect the result.
    """
    import_check = (
        "import sys\n"
        "import src.prediction\n"
        "import src.explainability\n"
        'assert "src.modeling" not in sys.modules\n'
        'assert "mlflow" not in sys.modules\n'
        'assert "dvc" not in sys.modules\n'
    )

    subprocess.run(
        [sys.executable, "-c", import_check],
        check=True,
    )
