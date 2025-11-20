"""Test integration between Rust backend and Python Array API."""

import numpy as np
import pytest

try:
    import lacuna.array_api as xp
    from lacuna import _core
    from lacuna.sparse import CSR

    HAS_CORE = True
except ImportError:
    HAS_CORE = False

pytestmark = pytest.mark.skipif(not HAS_CORE, reason="Rust core not available")


class TestCreationFunctions:
    """Test that creation functions use Rust backend."""

    def test_csr_zeros_basic(self):
        """Test CSR.zeros creates empty matrix."""
        A = CSR.zeros((10, 20))
        assert A.shape == (10, 20)
        assert A.nnz == 0
        assert len(A.indptr) == 11
        assert len(A.indices) == 0
        assert len(A.data) == 0

    def test_csr_eye_basic(self):
        """Test CSR.eye creates identity matrix."""
        I = CSR.eye(5)
        assert I.shape == (5, 5)
        assert I.nnz == 5
        assert I.sum() == 5.0

        # Check structure
        assert np.array_equal(I.indptr, [0, 1, 2, 3, 4, 5])
        assert np.array_equal(I.indices, [0, 1, 2, 3, 4])
        assert np.array_equal(I.data, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_csr_diag_basic(self):
        """Test CSR.diag creates diagonal matrix."""
        D = CSR.diag([1.0, 2.0, 3.0])
        assert D.shape == (3, 3)
        assert D.nnz == 3
        assert D.sum() == 6.0

        # Check structure
        assert np.array_equal(D.indptr, [0, 1, 2, 3])
        assert np.array_equal(D.indices, [0, 1, 2])
        assert np.array_equal(D.data, [1.0, 2.0, 3.0])

    def test_eye_matmul(self):
        """Test identity matrix multiplication."""
        I = CSR.eye(3)
        x = np.array([1.0, 2.0, 3.0])
        y = I @ x
        assert np.allclose(y, x)

    def test_diag_matmul(self):
        """Test diagonal matrix multiplication."""
        D = CSR.diag([2.0, 3.0, 4.0])
        x = np.array([1.0, 1.0, 1.0])
        y = D @ x
        assert np.allclose(y, [2.0, 3.0, 4.0])


class TestArrayAPIIntegration:
    """Test Array API layer uses Rust backend."""

    def test_array_api_zeros(self):
        """Test xp.zeros uses Rust CSR."""
        A = xp.zeros((10, 20))
        assert isinstance(A, CSR)
        assert A.nnz == 0
        assert hasattr(A, "_handle")
        if A._handle is not None:
            assert isinstance(A._handle, _core.Csr64)

    def test_array_api_eye(self):
        """Test xp.eye uses Rust CSR."""
        I = xp.eye(5)
        assert isinstance(I, CSR)
        assert I.nnz == 5
        assert I.sum() == 5.0
        assert hasattr(I, "_handle")
        if I._handle is not None:
            assert isinstance(I._handle, _core.Csr64)

    def test_array_api_eye_dtype(self):
        """Test xp.eye respects dtype."""
        I = xp.eye(3, dtype=np.float64)
        assert I.dtype == np.float64

    def test_array_api_zeros_shape(self):
        """Test xp.zeros with various shapes."""
        shapes = [(5, 5), (10, 20), (1, 100), (100, 1)]
        for shape in shapes:
            A = xp.zeros(shape)
            assert A.shape == shape
            assert A.nnz == 0

    def test_array_api_device_validation(self):
        """Test device parameter validation."""
        with pytest.raises(ValueError, match="cpu"):
            xp.zeros((10, 10), device="cuda")

        with pytest.raises(ValueError, match="cpu"):
            xp.eye(5, device="gpu")

    def test_array_api_eye_limitations(self):
        """Test current limitations of eye function."""
        # Non-square not supported yet
        with pytest.raises(NotImplementedError):
            xp.eye(5, n_cols=10)

        # Off-diagonal not supported yet
        with pytest.raises(NotImplementedError):
            xp.eye(5, k=1)


class TestRustBackendUsage:
    """Test that operations actually use Rust backend."""

    def test_zeros_uses_rust_function(self):
        """Verify zeros calls _core.zeros_csr."""
        # This implicitly tests that _core.zeros_csr exists and works
        A = CSR.zeros((5, 10))
        assert A._handle is not None

    def test_eye_uses_rust_function(self):
        """Verify eye calls _core.eye_csr."""
        I = CSR.eye(3)
        assert I._handle is not None

    def test_diag_uses_rust_function(self):
        """Verify diag calls _core.diag_csr."""
        D = CSR.diag([1.0, 2.0])
        assert D._handle is not None

    def test_operations_on_rust_created_matrices(self):
        """Test that Rust-created matrices work with operations."""
        I = CSR.eye(10)

        # Test sum
        assert I.sum() == 10.0

        # Test matmul
        x = np.ones(10)
        y = I @ x
        assert np.allclose(y, x)

        # Test transpose
        IT = I.T
        assert IT.shape == (10, 10)
        assert IT.nnz == 10


class TestFallbackBehavior:
    """Test fallback when Rust core is not available."""

    def test_zeros_fallback(self, monkeypatch):
        """Test that zeros has pure Python fallback."""
        # Temporarily disable _core
        import lacuna.sparse.csr as csr_module

        original_core = csr_module._core
        monkeypatch.setattr(csr_module, "_core", None)

        A = CSR.zeros((5, 10))
        assert A.shape == (5, 10)
        assert A.nnz == 0

        # Restore
        monkeypatch.setattr(csr_module, "_core", original_core)

    def test_eye_fallback(self, monkeypatch):
        """Test that eye has pure Python fallback."""
        import lacuna.sparse.csr as csr_module

        original_core = csr_module._core
        monkeypatch.setattr(csr_module, "_core", None)

        I = CSR.eye(3)
        assert I.shape == (3, 3)
        assert I.nnz == 3
        # Note: sum() requires Rust core, so we just check basic properties
        assert len(I.indptr) == 4
        assert len(I.indices) == 3
        assert len(I.data) == 3
        assert np.array_equal(I.data, [1.0, 1.0, 1.0])

        monkeypatch.setattr(csr_module, "_core", original_core)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
