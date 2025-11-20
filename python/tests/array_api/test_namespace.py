import lacuna.array_api as xp


def test_namespace_info_capabilities():
    info = xp.__array_namespace_info__()
    caps = info["capabilities"]
    assert caps["sparse"] is True
    # Check selected capabilities advertised
    assert "matmul" in caps.get("linalg", [])
    assert "matrix_transpose" in caps.get("linalg", [])
    assert "sum" in caps.get("reductions", [])
    assert "mean" in caps.get("reductions", [])
    assert "count_nonzero" in caps.get("reductions", [])
    assert "add" in caps.get("elementwise", [])
    assert "multiply" in caps.get("elementwise", [])
    assert "zeros" in caps.get("creation", [])
    assert "eye" in caps.get("creation", [])
    assert "permute_dims" in caps.get("manipulation", [])
    assert "reshape" in caps.get("manipulation", [])
