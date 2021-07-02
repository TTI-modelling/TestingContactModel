from unittest.mock import patch

import pytest

from household_contact_tracing.utilities import validate_sequences_length, process_linear_sequences, \
    process_combinatorial_sequences, process_sequences, ParameterError


class TestValidateSequences:
    def test_single_sequence(self):
        """If there is a single list the function should return True."""
        assert validate_sequences_length([[1, 2]]) is True

    def test_valid_sequences(self):
        """If there are lists of the same length the function should return True"""
        assert validate_sequences_length([[1, 2], [1, 2]]) is True

    def test_invalid_sequences(self):
        """Validate sequences should return False if lists are different lengths."""
        assert validate_sequences_length([[1, 2], [1, 2, 3]]) is False


@patch('household_contact_tracing.utilities.validate_sequences_length', return_value=True)
class TestProcessLinearSequences:
    def test_single_element(self, mock_validate_sequences):
        """A single sequence with a single item should return one set of params"""
        params = {"a": 1, "sequences": {"b": [2]}}
        expected_result = [{"a": 1, "b": 2}]
        assert process_linear_sequences(params) == expected_result

    def test_single_sequence(self, mock_validate_sequences):
        """A single sequence with multiple items should return multiple params"""
        params = {"a": 1, "sequences": {"b": [2, 3]}}
        expected_result = [{"a": 1, "b": 2}, {"a": 1, "b": 3}]
        assert process_linear_sequences(params) == expected_result

    def test_multiple_sequences(self, mock_validate_sequences):
        """Multiple sequences with multiple items should return multiple params with the
        sequences combined combinatorially."""
        params = {"a": 1, "sequences": {"b": [2, 3], "c": [2, 3]}}
        expected_result = [{"a": 1, "b": 2, "c": 2}, {"a": 1, "b": 3, "c": 3}]
        assert process_linear_sequences(params) == expected_result


class TestProcessCombinatorialSequences:
    def test_single_element(self):
        """A single sequence with a single item should return one set of params"""
        params = {"a": 1, "sequences": {"b": [2]}}
        expected_result = [{"a": 1, "b": 2}]
        assert process_combinatorial_sequences(params) == expected_result

    def test_single_sequence(self):
        """A single sequence with multiple items should return multiple params"""
        params = {"a": 1, "sequences": {"b": [2, 3]}}
        expected_result = [{"a": 1, "b": 2}, {"a": 1, "b": 3}]
        assert process_combinatorial_sequences(params) == expected_result

    def test_two_sequences(self):
        """Two sequences with multiple items should return all possible combinations
        of the sequences."""
        params = {"a": 1, "sequences": {"b": [2, 3], "c": [2, 3]}}
        expected_result = [{"a": 1, "b": 2, "c": 2},
                           {"a": 1, "b": 2, "c": 3},
                           {"a": 1, "b": 3, "c": 2},
                           {"a": 1, "b": 3, "c": 3}]
        assert process_combinatorial_sequences(params) == expected_result

    def test_three_sequences(self):
        """Three sequences with multiple items should return all possible combinations
        of the sequences."""
        params = {"a": 1, "sequences": {"b": [2, 3], "c": [2, 3], "d": [2, 3]}}
        expected_result = [{"a": 1, "b": 2, "c": 2, "d": 2},
                           {"a": 1, "b": 2, "c": 2, "d": 3},
                           {"a": 1, "b": 2, "c": 3, "d": 2},
                           {"a": 1, "b": 2, "c": 3, "d": 3},
                           {"a": 1, "b": 3, "c": 2, "d": 2},
                           {"a": 1, "b": 3, "c": 2, "d": 3},
                           {"a": 1, "b": 3, "c": 3, "d": 2},
                           {"a": 1, "b": 3, "c": 3, "d": 3}]
        assert process_combinatorial_sequences(params) == expected_result


@patch('household_contact_tracing.utilities.process_combinatorial_sequences')
@patch('household_contact_tracing.utilities.process_linear_sequences')
class TestProcessSequences:
    """Test that the right functions are called when sequences are provided in parameters."""

    def test_no_sequences(self, mock_process_combinatorial, mock_process_linear):
        """When no sequences are provided, a list containing the input should be provided.."""
        params = {"a": 1}
        result = process_sequences(params)
        assert not mock_process_linear.called
        assert not mock_process_combinatorial.called
        assert result == [params]

    def test_nesting_type_unspecified(self, mock_process_linear, mock_process_combinatorial):
        """When sequences are provided, and no nesting_type is specified, the default
        is to combine linearly."""
        params = {"a": 1, "sequences": {"a": 1}}
        process_sequences(params)
        assert mock_process_linear.called
        assert not mock_process_combinatorial.called

    def test_nesting_type_linear(self, mock_process_linear, mock_process_combinatorial):
        """When sequences are provided, and linear nesting_type is specified, the sequences are
        nested linearly."""
        params = {"a": 1, "sequences": {"nesting_type": "linear", "a": 1}}
        process_sequences(params)
        assert mock_process_linear.called
        assert not mock_process_combinatorial.called

    def test_nesting_type_combination(self, mock_process_linear, mock_process_combinatorial):
        """When sequences are provided, and a combination nesting_type is specified, the sequences
        are nested combinatorially."""
        params = {"a": 1, "sequences": {"nesting_type": "combination", "a": 1}}
        process_sequences(params)
        assert not mock_process_linear.called
        assert mock_process_combinatorial.called

    def test_invalid_nesting_type(self, mock_process_linear, mock_process_combinatorial):
        """When an invalid nesting type is specified, an error is raised."""
        params = {"a": 1, "sequences": {"nesting_type": "invalid_input", "a": 1}}
        with pytest.raises(ParameterError):
            process_sequences(params)
