import importlib
import importlib.util


def _comparison_module():
    module_name = "src.vlm.analyze_backend_comparison"
    assert importlib.util.find_spec(module_name) is not None, (
        f"{module_name} must exist"
    )
    return importlib.import_module(module_name)


def test_classify_pair_covers_all_outcomes() -> None:
    comparison = _comparison_module()

    assert comparison.classify_pair(1, 1) == "both_success"
    assert comparison.classify_pair(0, 1) == "cnn_only"
    assert comparison.classify_pair(1, 0) == "geometric_only"
    assert comparison.classify_pair(0, 0) == "both_failure"
