"""

Manage extracting configs for each package

Logically connected to cache_io but there are not shared imports

This is a bit out-of-hand, so see below:

-=-=-=- Functinality List -=-=-=-=

1.) another_config_with_filled_values = extract_example_config(some_config_for_some_exp)
    The idea is that the input config <some_config_for_some_exp> is
    used within an experiment.
    We don't want to modify the input config, but we want to enumerate
    all the parameters for a single config.

"""

from .econfig import ExtractConfig
