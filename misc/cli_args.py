# -*- coding: utf-8 -*-
# =============================================================================
# This file is part of WaterGAP.

# WaterGAP is an opensource software which computes water flows and storages as
# well as water withdrawals and consumptive uses on all continents.

# You should have received a copy of the LGPLv3 License along with WaterGAP.
# if not see <https://www.gnu.org/licenses/lgpl-3.0>
# =============================================================================

"""Arguments for command line interface (CLI)."""

# =============================================================================
# This module is used by Configuration handler and Input handler.
# =============================================================================


import argparse
import os


def parse_cli():
    """
    Parse command line arguments.

    In web mode (WATERGAP_CONFIG env var set), argparse is skipped
    and a compatible namespace object is returned instead.

    Returns
    -------
    Command line argument
    """
    # Web mode: skip argparse when WATERGAP_CONFIG is set
    config_path = os.environ.get('WATERGAP_CONFIG')
    if config_path is not None:
        debug = os.environ.get('WATERGAP_DEBUG', 'false').lower() == 'true'
        return argparse.Namespace(name=config_path, debug=debug)

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, metavar='',
                        help='name of configuration file',)
    parser.add_argument('--debug', action="store_true",
                        help='Enable or disable TraceBack for '
                        'debugging by setting True or False ')
    args = parser.parse_args()
    return args
