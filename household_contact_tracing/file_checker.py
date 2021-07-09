import os
import re
from loguru import logger


class FileChecker:
    """
        Base class containing logic for checking and assignment of system files
        Directory and filename validation
        Inherit from this base class if filename checking is required.

        Methods
        -------
            validate_filename(cls, filename)
                Class method: Validate filename
    """

    @classmethod
    def validate_file_spec(cls, file_spec: str) -> str:
        """
        Validate a file spec string.
            The directory must exist and the filename must be valid (does not have to exist)

            Parameters:
                file_spec (str): a file spec '/home/xyz/abc.csv'

            Returns:
                the valid file_spec, as input

            Exceptions
                ValueError
        """

        # Check directory exists
        if cls.dir_name_exists(file_spec):
            # Check file name is (without directory path) is valid.
            if cls.filename_valid(file_spec):
                return file_spec

    @classmethod
    def filename_valid(cls, file_spec: str) -> str:
        """
        Validate the filename part of an input file spec string.
        E.g. for '/home/xyz/abc.csv'  checks if abc.csv is a valid filename (does not have to exist)

            Parameters:
                file_spec (str): a file spec '/home/xyz/abc.csv'

            :return:
                the valid file_spec, as input

            :except
                ValueError

        """

        logger.debug("Checking filename is valid.")
        f_name = os.path.basename(file_spec)
        re_result = re.search(r'[^A-Za-z0-9_\-\.\\]', f_name)
        if re_result:
            logger.debug("Filename {} is invalid.".format(file_spec))
            raise ValueError('{} contains invalid characters for a filename'.format(file_spec))
        else:
            logger.debug("Filename successfully validated.")
            return file_spec

    @classmethod
    def dir_name_exists(cls, dir_or_filename: str) -> str:
        """
        "Check that the directory name or file name in input string exists.
        E.g. for '/home/xyz/abc.csv'  checks if directory /home/xyz exists

            Parameters:
                dir_or_filename (str): a file spec e.g. '/home/xyz/abc.csv'
                                        or a directory e.g. '/home/xyz'

            Returns:
                the valid dir_or_filename, as input

            Exceptions
                IsADirectoryError

        """
        logger.debug("Checking directory in input dir name or filename string exists.")

        if os.path.dirname(dir_or_filename) and os.path.exists(os.path.dirname(dir_or_filename)):
            return dir_or_filename
        else:
            logger.debug("File/dir name {} does not exist.".format(dir_or_filename))
            raise IsADirectoryError('Directory {} does not exist'.format(os.path.dirname(dir_or_filename)))
