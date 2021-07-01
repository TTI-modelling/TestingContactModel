from unittest.mock import patch

from household_contact_tracing.utilities import validate_sequences, process_combinatorial_sequences


class TestValidateSequences:
    def test_single_sequence(self):
        """If there is a single list the function should return True."""
        assert validate_sequences([[1, 2]]) is True

    def test_valid_sequences(self):
        """If there are lists of the same length the function should return True"""
        assert validate_sequences([[1, 2], [1, 2]]) is True

    def test_invalid_sequences(self):
        """Validate sequences should return False if lists are different lengths."""
        assert validate_sequences([[1, 2], [1, 2, 3]]) is False


@patch('household_contact_tracing.utilities.validate_sequences', return_value=True)
class TestProcessCombinatorialSequences:
    def test_single_element(self, mock_validate_sequences):
        """A single sequence with a single item should return one set of params"""
        params = {"a": 1, "sequences": {"b": [2]}}
        expected_result = [{"a": 1, "b": 2}]
        assert process_combinatorial_sequences(params) == expected_result

    def test_single_sequence(self, mock_validate_sequences):
        """A single sequence with a multiple items should return multiple params"""
        params = {"a": 1, "sequences": {"b": [2, 3]}}
        expected_result = [{"a": 1, "b": 2}, {"a": 1, "b": 3}]
        assert process_combinatorial_sequences(params) == expected_result

    def test_multiple_sequences(self, mock_validate_sequences):
        """A multiple sequences with a multiple items should return multiple params with the
        sequences combined combinatorially."""
        params = {"a": 1, "sequences": {"b": [2, 3], "c": [2, 3]}}
        expected_result = [{"a": 1, "b": 2, "c": 2}, {"a": 1, "b": 3, "c": 3}]
        assert process_combinatorial_sequences(params) == expected_result
