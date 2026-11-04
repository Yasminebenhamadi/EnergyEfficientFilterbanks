import librosa
import numpy as np
import cmsisdsp as dsp
from scipy.signal import windows
from numpy.lib.stride_tricks import sliding_window_view
LOG_EPS=1e-10


# For devices running CMSIS-DSP 

class MelSpecCMSIS:
    def __init__(
        self,
        samplerate: int,
        window_len: int,
        window_stride: int,
        nb_mels: int,
        fmin: float,
        fmax: float,
    ):
        """Initialize a CMSIS-DSP-compatible Mel-spectrogram frontend."""

        if window_len <= 0 or window_stride <= 0:
            raise ValueError("Window length and stride must be positive.")

        if not (0 <= fmin < fmax <= samplerate / 2):
            raise ValueError("Invalid Mel frequency range.")

        self.name = (
            f"MelSpecCMSIS_sr{samplerate}_win{window_len}"
            f"_hop{window_stride}_mels{nb_mels}"
            f"_fmin{fmin}_fmax{fmax}"
        )

        self.samplerate = samplerate
        self.window_len = window_len
        self.window_stride = window_stride
        self.nb_mels = nb_mels
        self.fmin = fmin
        self.fmax = fmax

        self.fft_instance = dsp.arm_rfft_fast_instance_f32()

        status = dsp.arm_rfft_fast_init_f32(
            self.fft_instance,
            window_len,
        )

        if status != 0:
            raise RuntimeError(
                f"CMSIS-DSP FFT initialization failed: {status}"
            )

        self.hann_window = windows.hann(window_len).astype(np.float32)

        (
            self.mel_filter_pos,
            self.mel_filter_len,
            self.mel_filter_coefs,
        ) = self._truncate_mels(
            samplerate, window_len, nb_mels, fmin, fmax
        )

    @staticmethod
    def _truncate_mels(samplerate, window_len, nb_mels, fmin, fmax):
        '''
        This gives truncated mel filterbanks that are meant to be applied per window
        i.e to a row of size (window_len/2 + 1,)
        '''
        mel_filter= librosa.filters.mel(sr=samplerate, n_fft=window_len, n_mels=nb_mels, fmin=fmin, fmax=fmax)

        mel_filter_pos = []
        mel_filter_len = []
        mel_filter_coefs = []

        for bin_idx, mel_bin in enumerate(mel_filter):

            nonzero = np.where(mel_bin > 0)[0]
            if len(nonzero) == 0:
                continue

            start_idx, end_idx = nonzero[0], nonzero[-1] + 1

            mel_filter_pos.append(start_idx)
            mel_filter_len.append(end_idx-start_idx)
            mel_filter_coefs.extend(mel_bin[start_idx:end_idx])

        assert(len(mel_filter_pos)==nb_mels)
        return mel_filter_pos, mel_filter_len, mel_filter_coefs
    
    def _apply_truncate_mels(self, window):

        '''
        Applies truncated mel filterbanks per window and returns (nb_mels, 1)
        '''

        mel_column = np.zeros((self.nb_mels))

        mel_idx=0
        
        for idx_mel in range(self.nb_mels):
            idx_win_start=self.mel_filter_pos[idx_mel]
            mel_size=self.mel_filter_len[idx_mel]
            trunc_window=window[idx_win_start:idx_win_start+mel_size]

            mel_coefs=self.mel_filter_coefs[mel_idx:mel_idx+mel_size]
            mel_idx=mel_idx+mel_size

            mel_column[idx_mel] = np.dot(trunc_window, mel_coefs)

        return mel_column

    def _process_window(self, window):
        hanned = dsp.arm_mult_f32(window, self.hann_window)

        # arm_rfft_fast_f32 returns packed nfft/2 elements layed out as [real(0), real(N/2), real(1), imag(1) .... real(N/2-1), imag(N/2-1)]
        fft = dsp.arm_rfft_fast_f32(self.fft_instance,
                                    hanned,
                                    0) # ifftFlag=0 for forward FFT else inverse fft
        
        # Expand back to nfft/2 + 1
        cmsis_cmplx_power = np.concatenate([
            [fft[0] * fft[0]],                 # DC
            dsp.arm_cmplx_mag_squared_f32(fft[2:]),
            [fft[1] * fft[1]]                  # Nyquist
        ]) 
        
        mel_spec = self._apply_truncate_mels(cmsis_cmplx_power)
        
        return dsp.arm_vlog_f32(mel_spec + LOG_EPS)

    def feature(self, audio, standard=True):
        """Extract log-Mel spectrogram features from an audio signal."""

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim != 1:
            raise ValueError("Expected a mono audio signal.")

        if len(audio) < self.window_len:
            raise ValueError("Audio signal is shorter than the FFT window.")

        windows_audio = sliding_window_view(
            audio, self.window_len
        )[::self.window_stride]

        outputs = [
            self._process_window(window)
            for window in windows_audio
        ]

        mel_spectrogram = np.asarray(outputs).T

        if standard:
            audio_std = np.std(audio)

            if audio_std <= 0:
                raise ValueError("Cannot normalize a silent audio signal.")

            mel_spectrogram -= 2 * np.log(audio_std)

        return mel_spectrogram