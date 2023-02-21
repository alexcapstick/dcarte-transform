"""
Progress functions and classes.
"""

import typing
import tqdm
import numpy as np

try:
    import pandarallel

    pandarallel_import_error = False
except ImportError:
    pandarallel_import_error = True

# package wide styling for progress bars
tqdm_style = {
    #'ascii':" ▖▘▝▗▚▞▉",
    "ascii": "▏▎▍▋▊▉",
    #'colour':'black',
    "dynamic_ncols": True,
}


# monkey patch the progress bar in
# https://github.com/nalepae/pandarallel/blob/bb0f50faf9bd3e8e548736040d297613d4482eaa/pandarallel/progress_bars.py#L19


class TQDMProgressBarPandarallelGenerator:
    def __init__(self, **tqdm_kwargs):
        if pandarallel_import_error:
            raise ImportError(
                "pandarallel not installed, please install it to use this class."
            )
            

        self.tqdm_kwargs = tqdm_kwargs
        return

    def get_bar(self):

        tqdm_kwargs = self.tqdm_kwargs

        # monkey patched class
        class TQDMProgressBarPandarallel(pandarallel.progress_bars.ProgressBars):
            def __init__(self, maxs: typing.List[int], show: bool) -> None:
                total = np.sum(maxs)
                self.tqdm_progress = tqdm.tqdm(
                    total=total,
                    disable=not show,
                    **tqdm_kwargs,
                )
                self.value = 0

            def update(self, values: typing.List[int]) -> None:
                """Update a bar value.
                Positional arguments:
                values - The new values of each bar
                """
                update_amount = np.sum(values) - self.value
                self.tqdm_progress.update(update_amount)
                self.tqdm_progress.refresh()
                self.value = np.sum(values)

        return TQDMProgressBarPandarallel


# funciton that replaces the pandarallel progress bars
def pandarallel_progress(**tqdm_kwargs):
    bar_generator = TQDMProgressBarPandarallelGenerator(**tqdm_kwargs)
    pandarallel.progress_bars.ProgressBarsNotebookLab = bar_generator.get_bar()
    pandarallel.progress_bars.ProgressBarsConsole = bar_generator.get_bar()
    return
