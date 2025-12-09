# pyre-strict

import torch
from pfns.priors.convert_prior_to_x_only_format import get_batch
from pfns.priors.prior import Batch


def create_simple_traditional_get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    single_eval_pos: int,
    hyperparameters: dict | None = None,
    n_targets_per_input: int = 1,
    **kwargs,
) -> Batch:
    """
    Simple traditional get_batch function for testing.
    Creates linear functions: y = sum(x) + noise
    """
    # Generate random input features
    x = torch.rand(batch_size, seq_len, num_features)

    # Create simple linear relationship: y = sum(x_features) + small noise
    y = x.sum(dim=2, keepdim=True) + torch.randn(batch_size, seq_len, 1) * 0.1

    # For traditional format, target_y is the same as y but potentially repeated
    target_y = y.repeat(1, 1, n_targets_per_input)

    return Batch(
        x=x,
        y=y,
        target_y=target_y,
        single_eval_pos=single_eval_pos,
    )


def create_complex_traditional_get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    single_eval_pos: int,
    hyperparameters: dict | None = None,
    n_targets_per_input: int = 1,
    **kwargs,
) -> Batch:
    """
    More complex traditional get_batch function with optional attributes for testing.
    """
    x = torch.rand(batch_size, seq_len, num_features)
    y = x.mean(dim=2, keepdim=True) + torch.randn(batch_size, seq_len, 1) * 0.2
    target_y = y.repeat(1, 1, n_targets_per_input)

    # Add some optional attributes
    style = torch.randn(batch_size, 3)  # 3 style dimensions
    y_style = torch.randn(batch_size, 2)  # 2 y_style dimensions

    return Batch(
        x=x,
        y=y,
        target_y=target_y,
        single_eval_pos=single_eval_pos,
        style=style,
        y_style=y_style,
    )


class TestConvertPriorToXOnlyFormat:
    """Test suite for the convert_prior_to_x_only_format wrapper function."""

    def test_basic_conversion_functionality(self) -> None:
        """Test that the basic conversion from traditional to x-only format works correctly."""
        batch_size = 4
        seq_len = 10
        num_features = 3
        single_eval_pos = 6

        x_only_batch = get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            get_batch=create_simple_traditional_get_batch,
        )

        # Check that x-only format fields are properly set
        assert x_only_batch.y is None
        assert x_only_batch.target_y is None
        assert x_only_batch.x is not None
        assert x_only_batch.test_x is not None
        assert x_only_batch.target is not None

        # Check shapes
        assert x_only_batch.x.shape == (batch_size, single_eval_pos, num_features + 1)
        assert x_only_batch.test_x.shape == (
            batch_size,
            seq_len - single_eval_pos,
            num_features + 1,
        )
        assert x_only_batch.target.shape == (
            batch_size,
            seq_len - single_eval_pos,
            num_features + 1,
        )

    def test_x_format_contains_concatenated_features_and_y(self) -> None:
        """Test that x contains the concatenation of training features and training y values."""
        batch_size = 2
        seq_len = 8
        num_features = 2
        single_eval_pos = 5

        # Create traditional batch first to compare
        traditional_batch = create_simple_traditional_get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
        )

        x_only_batch = get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            get_batch=create_simple_traditional_get_batch,
        )

        # Extract training portions
        x_train_traditional = traditional_batch.x[:, :single_eval_pos, :]
        y_train_traditional = traditional_batch.y[:, :single_eval_pos, :]

        # Check that x in x-only format is [x_features, y_values]
        expected_x = torch.cat([x_train_traditional, y_train_traditional], dim=2)
        torch.testing.assert_close(x_only_batch.x, expected_x)

    def test_test_x_has_nan_for_y_values(self) -> None:
        """Test that test_x contains NaN for y values (to be predicted)."""
        batch_size = 3
        seq_len = 7
        num_features = 2
        single_eval_pos = 4

        x_only_batch = get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            get_batch=create_simple_traditional_get_batch,
        )

        # Check that the last column (y values) in test_x contains only NaN
        y_column_in_test_x = x_only_batch.test_x[:, :, -1]
        assert torch.all(torch.isnan(y_column_in_test_x))

        # Check that the feature columns don't contain NaN
        feature_columns_in_test_x = x_only_batch.test_x[:, :, :-1]
        assert not torch.any(torch.isnan(feature_columns_in_test_x))

    def test_target_contains_features_and_target_y(self) -> None:
        """Test that target contains test features concatenated with target y values."""
        batch_size = 2
        seq_len = 9
        num_features = 3
        single_eval_pos = 5

        # Create traditional batch to compare
        traditional_batch = create_simple_traditional_get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
        )

        x_only_batch = get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            get_batch=create_simple_traditional_get_batch,
        )

        # Extract test portions from traditional format
        x_test_traditional = traditional_batch.x[:, single_eval_pos:, :]
        target_y_test_traditional = traditional_batch.target_y[:, single_eval_pos:, :]

        # Check that target in x-only format is [x_test_features, target_y_values]
        expected_target = torch.cat(
            [x_test_traditional, target_y_test_traditional], dim=2
        )
        torch.testing.assert_close(x_only_batch.target, expected_target)

    def test_multiple_targets_per_input(self) -> None:
        """Test conversion with n_targets_per_input > 1."""
        batch_size = 2
        seq_len = 8
        num_features = 2
        single_eval_pos = 5
        n_targets_per_input = 3

        x_only_batch = get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            get_batch=create_simple_traditional_get_batch,
            n_targets_per_input=n_targets_per_input,
        )

        # With multiple targets, the target shape should be expanded
        expected_target_shape = (
            batch_size,
            seq_len - single_eval_pos,
            num_features + 1,
            n_targets_per_input,
        )
        assert x_only_batch.target.shape == expected_target_shape

        # x and test_x should still have the same shapes as single target case
        assert x_only_batch.x.shape == (batch_size, single_eval_pos, num_features + 1)
        assert x_only_batch.test_x.shape == (
            batch_size,
            seq_len - single_eval_pos,
            num_features + 1,
        )

    def test_preserves_optional_attributes(self) -> None:
        """Test that optional attributes from the traditional batch are preserved."""
        batch_size = 2
        seq_len = 6
        num_features = 2
        single_eval_pos = 3

        x_only_batch = get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            get_batch=create_complex_traditional_get_batch,
        )

        # Check that optional attributes are preserved
        assert x_only_batch.style is not None
        assert x_only_batch.y_style is not None
        assert x_only_batch.style.shape == (batch_size, 3)
        assert x_only_batch.y_style.shape == (batch_size, 2)
        assert x_only_batch.single_eval_pos == single_eval_pos

    def test_hyperparameters_passed_through(self) -> None:
        """Test that hyperparameters are correctly passed to the wrapped function."""

        def test_get_batch_with_hyperparams(
            batch_size: int,
            seq_len: int,
            num_features: int,
            single_eval_pos: int,
            hyperparameters: dict | None = None,
            **kwargs,
        ) -> Batch:
            # Check that hyperparameters are passed correctly
            assert hyperparameters is not None
            assert hyperparameters["test_param"] == "test_value"

            return create_simple_traditional_get_batch(
                batch_size,
                seq_len,
                num_features,
                single_eval_pos,
                hyperparameters,
                **kwargs,
            )

        test_hyperparams = {"test_param": "test_value"}

        x_only_batch = get_batch(
            batch_size=2,
            seq_len=6,
            num_features=2,
            single_eval_pos=3,
            get_batch=test_get_batch_with_hyperparams,
            hyperparameters=test_hyperparams,
        )

        # If we get here without assertion error, hyperparameters were passed correctly
        assert x_only_batch is not None

    def test_kwargs_passed_through(self) -> None:
        """Test that additional kwargs are correctly passed to the wrapped function."""

        def test_get_batch_with_kwargs(
            batch_size: int,
            seq_len: int,
            num_features: int,
            single_eval_pos: int,
            hyperparameters: dict | None = None,
            extra_param: str = "default",
            **kwargs,
        ) -> Batch:
            # Check that kwargs are passed correctly
            assert extra_param == "test_extra"

            return create_simple_traditional_get_batch(
                batch_size,
                seq_len,
                num_features,
                single_eval_pos,
                hyperparameters,
                **kwargs,
            )

        x_only_batch = get_batch(
            batch_size=2,
            seq_len=6,
            num_features=2,
            single_eval_pos=3,
            get_batch=test_get_batch_with_kwargs,
            extra_param="test_extra",
        )

        # If we get here without assertion error, kwargs were passed correctly
        assert x_only_batch is not None

    def test_empty_hyperparameters_default(self) -> None:
        """Test that empty hyperparameters dict is used when None is passed."""

        def test_get_batch_checks_empty_hyperparams(
            batch_size: int,
            seq_len: int,
            num_features: int,
            single_eval_pos: int,
            hyperparameters: dict | None = None,
            **kwargs,
        ) -> Batch:
            # When hyperparameters=None is passed to wrapper, it should become {}
            assert hyperparameters == {}

            return create_simple_traditional_get_batch(
                batch_size,
                seq_len,
                num_features,
                single_eval_pos,
                hyperparameters,
                **kwargs,
            )

        x_only_batch = get_batch(
            batch_size=2,
            seq_len=6,
            num_features=2,
            single_eval_pos=3,
            get_batch=test_get_batch_checks_empty_hyperparams,
            hyperparameters=None,
        )

        assert x_only_batch is not None

    def test_single_eval_pos_boundary_conditions(self) -> None:
        """Test behavior at boundary conditions for single_eval_pos."""
        batch_size = 2
        seq_len = 6
        num_features = 2

        # Test with single_eval_pos = 1 (minimal training data)
        x_only_batch = get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=1,
            get_batch=create_simple_traditional_get_batch,
        )

        assert x_only_batch.x.shape == (batch_size, 1, num_features + 1)
        assert x_only_batch.test_x.shape == (batch_size, seq_len - 1, num_features + 1)
        assert x_only_batch.target.shape == (batch_size, seq_len - 1, num_features + 1)

        # Test with single_eval_pos = seq_len - 1 (minimal test data)
        x_only_batch = get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=seq_len - 1,
            get_batch=create_simple_traditional_get_batch,
        )

        assert x_only_batch.x.shape == (batch_size, seq_len - 1, num_features + 1)
        assert x_only_batch.test_x.shape == (batch_size, 1, num_features + 1)
        assert x_only_batch.target.shape == (batch_size, 1, num_features + 1)

    def test_device_consistency(self) -> None:
        """Test that tensors maintain device consistency."""

        def get_batch_with_specific_device(
            batch_size: int,
            seq_len: int,
            num_features: int,
            single_eval_pos: int,
            hyperparameters: dict | None = None,
            **kwargs,
        ) -> Batch:
            device = torch.device("cpu")  # Force CPU for testing
            x = torch.rand(batch_size, seq_len, num_features, device=device)
            y = torch.rand(batch_size, seq_len, 1, device=device)
            target_y = y.clone()

            return Batch(x=x, y=y, target_y=target_y, single_eval_pos=single_eval_pos)

        x_only_batch = get_batch(
            batch_size=2,
            seq_len=6,
            num_features=2,
            single_eval_pos=3,
            get_batch=get_batch_with_specific_device,
        )

        # All tensors should be on the same device
        assert x_only_batch.x.device == x_only_batch.test_x.device
        assert x_only_batch.x.device == x_only_batch.target.device


if __name__ == "__main__":
    # Run a simple test if executed directly
    test_instance = TestConvertPriorToXOnlyFormat()
    test_instance.test_basic_conversion_functionality()
    print("Basic test passed!")
