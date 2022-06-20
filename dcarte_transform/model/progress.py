import typing
import pytorch_lightning as pl

#import importlib
#if importlib.util.find_spec("ipywidgets") is not None:
#    from tqdm.auto import tqdm as _tqdm
#else:
#    from tqdm import tqdm as _tqdm
from tqdm import tqdm as _tqdm
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar, convert_inf
import sys
from ..utils.progress import tqdm_style




# pytorch lightning progress bars

_PAD_SIZE = 5
# from 
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/callbacks/progress/tqdm_progress.py
class Tqdm(_tqdm):
    def __init__(self, *args, **kwargs) -> None:
        """Custom tqdm progressbar where we append 0 to floating points/strings to prevent the progress bar from
        flickering."""
        # this just to make the make docs happy, otherwise it pulls docs which has some issues...
        super().__init__(*args, **kwargs)

    @staticmethod
    def format_num(n: typing.Union[int, float, str]) -> str:
        """Add additional padding to the formatted numbers."""
        should_be_padded = isinstance(n, (float, str))
        if not isinstance(n, str):
            n = _tqdm.format_num(n)
            assert isinstance(n, str)
        if should_be_padded and "e" not in n:
            if "." not in n and len(n) < _PAD_SIZE:
                try:
                    _ = float(n)
                except ValueError:
                    return n
                n += "."
            n += "0" * (_PAD_SIZE - len(n))
        return n


class MyProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__()
        self._refresh_rate = self._resolve_refresh_rate(refresh_rate)
        self._process_position = process_position

    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        bar = Tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            file=sys.stdout,
            **tqdm_style,
        )
        return bar

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(
            desc=self.train_description,
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            file=sys.stdout,
            smoothing=0,
            **tqdm_style,
        )
        return bar

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = Tqdm(
            desc=self.predict_description,
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            file=sys.stdout,
            smoothing=0,
            **tqdm_style,
        )
        return bar

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=True, #self.is_disabled,
            leave= not has_main_bar,
            file=sys.stdout,
            smoothing=0,
            **tqdm_style,
        )
        return bar

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: typing.Any) -> None:
        super().on_train_epoch_start(trainer, *_)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch+1}")

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = Tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            file=sys.stdout,
            **tqdm_style,
        )
        return bar
