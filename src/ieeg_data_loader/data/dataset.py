import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, filtfilt, hilbert, firwin
import pickle
from pathlib import Path
import seaborn as sns
import os

from ..utils import non_overlapping_moving_average_repeated
from ..visualization import plot_continuous_heatmap, plot_continuous_signal, plot_eeg_bands


class IEEGData:
    def __init__(self, data=None, fs=None):
        self.trial_time_annotation = None
        self.time_annotation = None
        self.continuous_signal = None
        self.continuous_indicator = None
        self.data = data
        self.channel_name = None
        self.trial_info = None
        self.data_info = None
        self.time = None
        self.label = None
        self.fs = fs
        self.meta = None

    def dict_to_iEEG_format(self, data_dict, meta):
        self.data = data_dict['data']
        self.fs = data_dict['fs']
        self.time = data_dict['time']
        self.channel_name = data_dict['channel_names']
        self.trial_info = data_dict['trial_info']
        self.label = pd.DataFrame(data_dict['label'])
        self.meta = meta

    def epoched_to_continuous(self, patient, task, debug=False, save_path=None):
        """

        Args:
            settings:
            debug:
            save_path:

        Returns:

        """
        y = self.label['ColorLev'].values
        time = self.time * 1000
        n_trials, n_channels, _ = self.data.shape

        self.trial_info = self.trial_info.loc[:, ~self.trial_info.columns.duplicated()]

        if self.trial_time_annotation is None:
            self.get_time_annotation(patient=patient, task=task)
        continuous_signal = self.data[0]
        continuous_indicator = self._create_trial_indicator(task=task,
                                                            label=y,
                                                            trial_idx=0)
        continuous_indicator_list = [continuous_indicator]
        time_vecror = []
        for i in tqdm(range(n_trials - 1)):
            trial = self.data[i + 1]
            trial_indicator = self._create_trial_indicator(task=task,
                                                           label=y,
                                                           trial_idx=i + 1)
            for t in range(1, trial.shape[-1]):
                if np.all(continuous_signal[[0, 1, 10], -t:] == trial[[0, 1, 10], :t]):
                    # print(10000 - t)
                    time_vecror.append(10000 - t)
                    if debug is True:
                        if np.sum(continuous_signal[:, -t] - trial[:, 0]) != 0:
                            print(
                                f"Misalignment from trial {i} to {i + 1} in channels "
                                f"{np.where(continuous_signal[:, -t] != trial[:, 0])[0]}")
                    continuous_signal = np.concatenate((continuous_signal[:, :-t], trial), axis=1)
                    continuous_indicator = np.concatenate((
                        np.zeros_like(continuous_indicator[:-t]) * np.nan,
                        trial_indicator), axis=0)
                    continuous_indicator_list.append(continuous_indicator)
                    break
            if t == trial.shape[-1] - 1:
                print(f"error: trial {i} to {i + 1} alignment not found")
            all_indicator = [
                list(continuous_indicator) + [np.nan] * (continuous_signal.shape[-1] - len(continuous_indicator)) for
                continuous_indicator in continuous_indicator_list]

        all_indicator = np.stack(all_indicator)
        fig, ax = plt.subplots(1, figsize=(12, 10))
        sns.heatmap(all_indicator, ax=ax)

        if save_path is None:
            plt.show()
        else:
            fig.savefig(save_path + f"continuous_signals_annotation.png")
        final_indicator = np.ones_like(all_indicator[0]) * np.nan
        for t in tqdm(range(all_indicator.shape[-1])):
            if any(all_indicator[:, t] == 0):
                final_indicator[t] = 0
            elif len(np.unique(all_indicator[:, t])) == 2:
                if np.unique(all_indicator[:, t])[0] is not np.nan:
                    final_indicator[t] = np.unique(all_indicator[:, t])[0]
                else:
                    final_indicator[t] = np.unique(all_indicator[:, t])[1]
            elif len(np.unique(all_indicator[:, t])) > 2:
                trial_idxs = []
                for uniqevalue in list(np.unique(all_indicator[:, t])):
                    if uniqevalue is not np.nan:
                        trial_idxs.extend(list(np.where(all_indicator[:, t] == uniqevalue)[0]))
                final_indicator[t] = all_indicator[np.max(trial_idxs), t]

        self.continuous_indicator = final_indicator
        self.continuous_signal = continuous_signal

        if debug is True:
            diff = np.diff(final_indicator)
            change_indices = np.where(diff != 0)[0]
            change_indices += 1
            start_points = [idx for idx in change_indices if final_indicator[idx - 1] == 0]
            print(np.diff(start_points))

        return continuous_signal, final_indicator

    def get_time_annotation(self, patient, task):
        if task == 'imagine':
            # Get columns with duplicated names
            self.trial_info = self.trial_info.iloc[:, ~self.trial_info.columns.duplicated(keep='first')]

            imagination_period_end = (
                    self.trial_info['Onset Choice Period'] - self.trial_info['Fixation Onset NEV']).values
            trial_end = (self.trial_info['Trial offset'] - self.trial_info['Fixation Onset NEV']).values
            choice_period_end = (self.trial_info['Image Onset NEV'] - self.trial_info['Fixation Onset NEV']).values
            reaction_time = (self.trial_info['Response time NEV'] - self.trial_info['Fixation Onset NEV']).values
            if any(imagination_period_end < 1) or any(imagination_period_end > 7):
                imagination_period_end = (self.trial_info['CrossStart'] - self.trial_info['StartTrial']).values
            if any(np.isnan(trial_end)) or any(trial_end < 0):
                trial_end = (self.trial_info['TimeTrial'] - self.trial_info['StartTrial']).values
            if any(np.isnan(choice_period_end)):
                choice_period_end[np.isnan(choice_period_end)] = 5.5
                print("Warning: NaN in Image Onset NEV")

            self.time_annotation = {
                "imagination onset": 0,
                "choice period onset": np.round(np.mean(imagination_period_end) * 1000),
                "image onset": np.round(np.mean(choice_period_end) * 1000),
                "trial end": np.round(np.mean(trial_end) * 1000)
            }

            self.trial_time_annotation = {
                'imagination_period_end': imagination_period_end * 1000,
                'reaction_time': reaction_time * 1000,
                'choice_period_end': choice_period_end * 1000,
                'trial end': trial_end * 1000
            }
        elif task == 'flicker':
            fixation_onset = self.trial_info['Fixation Onset NEV'] - self.trial_info['Image Onset NEV']
            if 'Image Offset NEV' not in self.trial_info.columns:
                self.trial_info['Image Offset NEV'] = 0.0
                self.trial_info['Image Offset NEV'].values[:-1] = self.trial_info['Fixation Onset NEV'].values[1:]
            trial_end = self.trial_info['Image Offset NEV'] - self.trial_info['Image Onset NEV']

            if any(trial_end < 0.1) or any(trial_end > 5):
                print(
                    f"trials with trial_end annotation problem {np.where((trial_end < 0.1) | (trial_end > 5))[0]}")
                trial_end[(trial_end < 0.1) | (trial_end > 5)] = 1

            image_onset = np.zeros(self.trial_info.shape[0])
            if patient in ['p01', 'p03', 'p04']:
                trial_end = np.zeros(self.trial_info.shape[0])
                image_onset = -self.trial_info['TimeTrial'] + self.trial_info['TimeImageOnset']
                fixation_onset = image_onset - 0.5

            self.time_annotation = {
                "fixation onset": np.round(np.mean(fixation_onset - image_onset) * 1000),
                "image onset": np.round(0.0),
                "trial end": np.round(np.mean(trial_end - image_onset) * 1000)
            }

            self.trial_time_annotation = {
                "fixation onset": np.round(fixation_onset * 1000),
                "image onset": np.round(image_onset * 1000),
                "trial end": np.round(trial_end * 1000)
            }
        elif task == 'flicker_shape':
            image_onset = pd.DataFrame(np.zeros(self.trial_info.shape[0]))[0]
            if patient in ['p10']:
                trial_end = self.trial_info['Image Offset NEV'] - self.trial_info['Image Onset NEV']
                fixation_onset = -trial_end
                if any(fixation_onset > -0.2) or any(fixation_onset < -5):
                    idx_error = list(np.where((fixation_onset > -0.2) | (fixation_onset < -5))[0])
                    print(f"\nError: trials with fixation_onset annotation trial {idx_error} "
                          f"from {fixation_onset[idx_error[0]]} to {trial_end[idx_error[0]]}")
                    if (self.trial_info['Image Onset NEV'][idx_error[0]] - self.trial_info['Image Offset NEV'][
                        idx_error[0] - 1]) < 5:
                        trial_end[idx_error[0]] = self.trial_info['Image Onset NEV'][idx_error[0]] - \
                                                  self.trial_info['Image Offset NEV'][idx_error[0] - 1]
                        fixation_onset[idx_error[0]] = -trial_end[idx_error[0]]
                        print(f"Corrected from {fixation_onset[idx_error[0]]} to {trial_end[idx_error[0]]}\n")
                    else:
                        trial_end[idx_error[0]] = 1
                        fixation_onset[idx_error[0]] = -1
                        print(f"Corrected from {fixation_onset[idx_error[0]]} to {trial_end[idx_error[0]]}\n")
            elif patient in ['p11', 'p14']:
                print("This patient is aligned with fixation onset")
                fixation_onset = pd.DataFrame(np.zeros(self.trial_info.shape[0]))[0]
                image_onset = self.trial_info['Image Offset NEV'] - self.trial_info['Image Onset NEV']
                trial_end = 2 * image_onset
                if any(trial_end > 5) or any(trial_end < 0.1):
                    idx_error = list(np.where((trial_end > 5) | (trial_end < 0.1))[0])
                    print(f"==========\n"
                          f"Error: trials with fixation_onset annotation trial {idx_error} "
                          f"from {fixation_onset[idx_error[0]]} to {trial_end[idx_error[0]]}"
                          f"\n==========")
                    trial_end[idx_error[0]] = 2
                    image_onset[idx_error[0]] = 1
                    print(f"Corrected from {fixation_onset[idx_error[0]]} to {trial_end[idx_error[0]]}\n")
            elif patient in ['p12', 'p13', 'p15', 'p16', 'p17', 'p18']:
                fixation_onset = self.trial_info['Fixation Onset NEV'] - self.trial_info['Image Onset NEV']
                trial_end = -fixation_onset
                if any(fixation_onset > 0) or any(fixation_onset < -5):
                    print(
                        f"trials with fixation_onset annotation problem {np.where((fixation_onset > 0) | (fixation_onset < -5))[0]}")
                    trial_end[(fixation_onset > 0) | (fixation_onset < -5)] = -1
                if any(trial_end < 0.1) or any(trial_end > 5):
                    print(
                        f"trials with trial_end annotation problem {np.where((trial_end < 0.1) | (trial_end > 5))[0]}")
                    trial_end[(trial_end < 0.1) | (trial_end > 5)] = 1
            else:
                fixation_onset = self.trial_info['Fixation Onset NEV'] - self.trial_info['Image Onset NEV']
                trial_end = self.trial_info['Image Offset NEV'] - self.trial_info['Image Onset NEV']

            if any(trial_end > 5):
                print(f"trials with annotation problem {np.where(trial_end > 5)[0]}")
                trial_end[trial_end > 5] = 1

            self.time_annotation = {
                "fixation onset": np.round(np.mean(fixation_onset) * 1000),
                "image onset": np.round(np.mean(image_onset) * 1000),
                "trial end": np.round(np.mean(trial_end) * 1000)
            }

            self.trial_time_annotation = {
                "fixation onset": np.round(fixation_onset.values * 1000),
                "image onset": np.round(image_onset.values * 1000),
                "trial end": np.round(trial_end.values * 1000)
            }
        print(f"task {task} : \n {self.time_annotation}")

    def _create_trial_indicator(self, task, label, trial_idx):
        time = self.time * 1000
        continuous_indicator = np.zeros_like(self.time) * np.nan

        if task == 'imagine':
            print()
            imagination_period_onset = np.argmin(np.abs(time))
            choice_period_onset = np.argmin(
                np.abs(time - self.trial_time_annotation['imagination_period_end'][trial_idx]))
            reaction_time_onset = np.argmin(np.abs(time - self.trial_time_annotation['reaction_time'][trial_idx]))
            image_onset = np.argmin(np.abs(time - self.trial_time_annotation['choice_period_end'][trial_idx]))
            trial_end = np.argmin(np.abs(time - self.trial_time_annotation['trial_end'][trial_idx]))

            continuous_indicator[imagination_period_onset:choice_period_onset] = 2
            continuous_indicator[choice_period_onset:image_onset] = 3
            if reaction_time_onset != 0:
                continuous_indicator[reaction_time_onset] = 4
            continuous_indicator[image_onset:trial_end] = 2 * label[trial_idx] - 1
        elif task == 'flicker_shape':
            image_onset = np.argmin(np.abs(time - self.trial_time_annotation['image onset'][trial_idx]))
            fixation_onset = np.argmin(np.abs(time - self.trial_time_annotation['fixation onset'][trial_idx]))
            trial_end = np.argmin(np.abs(time - self.trial_time_annotation['trial end'][trial_idx]))

            continuous_indicator[fixation_onset:image_onset] = 0
            continuous_indicator[image_onset:trial_end] = 2 * label[trial_idx] - 1
        elif task == 'flicker':
            image_onset = np.argmin(np.abs(time - self.trial_time_annotation['image onset'][trial_idx]))
            fixation_onset = np.argmin(np.abs(time - self.trial_time_annotation['fixation onset'][trial_idx]))
            trial_end = np.argmin(np.abs(time - self.trial_time_annotation['trial end'][trial_idx]))

            continuous_indicator[fixation_onset:image_onset] = 0
            continuous_indicator[image_onset:trial_end] = 2 * label[trial_idx] - 1
        else:
            raise ValueError("task not defined")

        return continuous_indicator

    def normalize_data(self, signal, moving_avg_size=0):
        min_val = np.quantile(signal, 0.025, axis=-1, keepdims=True)
        max_val = np.quantile(signal, 0.975, axis=-1, keepdims=True)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        normalized_signal[normalized_signal > 1] = 1
        normalized_signal[normalized_signal < 0] = 0

        if moving_avg_size > 0:
            window = np.ones(moving_avg_size) / moving_avg_size
            normalized_signal = np.apply_along_axis(non_overlapping_moving_average_repeated, 1, normalized_signal,
                                                    moving_avg_size)
        return normalized_signal, min_val, max_val

    def get_trial_data(self, patient, start_time=-500, end_time=2000):
        y = self.label
        time = self.time*1000
        idx_start = np.argmin(abs(time - start_time))
        idx_end = np.argmin(abs(time - end_time))

        x = self.data[:, :, idx_start:idx_end]


        print(f"===========================\n"
              f"Time from {time[idx_start]} to {time[idx_end]}\n"
              f"===========================")
        idx_onset_list = []
        if patient in ['p01', 'p03', 'p04']:
            """
            for p01 we are assuming that times are not aligned with image onset but fixation onset
            We are keeping timings inside dataset.trial_time_annotation which contains 
            'fixation onset', 'image onset', 'trial end'
            x is the matrix of features and time_full_res is it's downsampled time vector
            """
            for trial_idx in range(x.shape[0]):
                idx_onset = np.argmin(abs(time - self.trial_time_annotation['image onset'][trial_idx]))

                if idx_onset == 0 and trial_idx == x.shape[0] - 1:
                    idx_onset = int(np.mean(idx_onset_list))
                else:
                    idx_start = np.argmin(
                        abs(time - (self.trial_time_annotation['image onset'][trial_idx] + start_time)))
                    idx_end = np.argmin(
                        abs(time - (self.trial_time_annotation['image onset'][trial_idx] + end_time)))
                idx_onset_list.append(idx_onset)
                if self.data[trial_idx, :, idx_start:idx_end].shape[-1] == x.shape[-1] - 1:
                    x[trial_idx, :, :] = self.data[trial_idx, :, idx_start:idx_end + 1]
                elif self.data[trial_idx, :, idx_start:idx_end].shape[-1] == x.shape[-1] + 1:
                    x[trial_idx, :, :] = self.data[trial_idx, :, idx_start:idx_end - 1]
                else:
                    x[trial_idx, :, :] = self.data[trial_idx, :, idx_start:idx_end]
            time = time[idx_start:idx_end]
            self.trial_time_annotation['trial end'] = - self.trial_time_annotation['image onset']
            self.trial_time_annotation['image onset'] = self.trial_time_annotation['image onset'] * 0
            self.trial_time_annotation['fixation onset'] = self.trial_time_annotation['image onset'] * 0 - 500
        return x, y, time

    @staticmethod
    def fir_bandpass(lowcut, highcut, fs, numtaps=101):
        nyq = 0.5 * fs
        taps = firwin(numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False)
        return taps

    def get_continuous_features(self, channel_number, settings, save_path=None, debug=False, normalize_signal=True):
        if self.continuous_signal is None:
            self.epoched_to_continuous(settings, debug=debug, save_path=save_path)

        # Downsample continuous_indicator
        downsample_factor_2 = int(self.fs / 250)
        continuous_indicator_downsampled = self.continuous_indicator[::downsample_factor_2]
        time = np.arange(0, continuous_indicator_downsampled.shape[-1]) / 250

        # Filter for 0.3Hz to 20Hz
        # Design the FIR filter
        taps = self.fir_bandpass(0.3, 20, self.fs, numtaps=101)
        signal_1 = filtfilt(taps, [1.0], self.continuous_signal, axis=1)

        # plot_freq_response_fir(taps, fs=self.fs)

        # Downsample signal_1 to 250Hz
        downsample_factor_1 = int(self.fs / 250)
        signal_erp_downsampled = signal_1[:, ::downsample_factor_1]

        if normalize_signal is True:
            signal_erp_downsampled, *_ = self.normalize_data(signal=signal_erp_downsampled, moving_avg_size=10)

        plot_continuous_heatmap(time=time,
                                signal=signal_erp_downsampled,
                                indicator=continuous_indicator_downsampled,
                                channel_names=self.channel_name,
                                normalize=normalize_signal,
                                cmap='jet',
                                save_path=save_path,
                                file_name='continuous_erp_heatmap')
        plot_continuous_signal(continuous_signal=signal_erp_downsampled,
                               continuous_indicator=continuous_indicator_downsampled,
                               channel_names=self.channel_name,
                               channel_number=channel_number,
                               settings=settings,
                               moving_avg_win=0,
                               save_path=save_path,
                               file_name='continuous_erp_signal')

        # Filter for 65Hz to 115Hz
        taps = self.fir_bandpass(65, 119, self.fs, numtaps=120)
        signal_2 = filtfilt(taps, [1.0], self.continuous_signal, axis=1)

        # plot_multitaper_psd(self.continuous_signal[0], fs=self.fs, NW=10)
        # plot_multitaper_psd(signal_2[0], fs=self.fs, NW=10)

        # Apply Hilbert transform
        signal_2_hilbert = np.abs(hilbert(signal_2, axis=1))

        # Downsample signal_2_hilbert to 250Hz
        downsample_factor_2 = int(self.fs / 250)
        signal_hga_downsampled = signal_2_hilbert[:, ::downsample_factor_2]

        # plot_multitaper_psd(signal_2_hilbert[0], fs=self.fs, NW=10)

        if normalize_signal is True:
            signal_hga_downsampled, *_ = self.normalize_data(signal=signal_hga_downsampled, moving_avg_size=10)

        plot_continuous_heatmap(time=time,
                                signal=signal_hga_downsampled,
                                indicator=continuous_indicator_downsampled,
                                channel_names=self.channel_name,
                                cmap='jet',
                                normalize=normalize_signal,
                                save_path=save_path,
                                file_name='continuous_hga_heatmap')
        plot_continuous_signal(continuous_signal=signal_hga_downsampled,
                               continuous_indicator=continuous_indicator_downsampled,
                               channel_names=self.channel_name,
                               channel_number=channel_number,
                               settings=settings,
                               moving_avg_win=50,
                               save_path=save_path,
                               file_name='continuous_hga_signal')

        return signal_erp_downsampled, signal_hga_downsampled

    def get_trial_features(self, data, lowcut, highcut, fs, fs_downsample, normalize_signal=False, moving_avg_size=0,
                           enable_hilbert=True):

        downsample_factor = int(self.fs / fs_downsample)
        time_downsampled = self.time[::downsample_factor]

        # Create bandpass filter taps
        taps = self.fir_bandpass(lowcut, highcut, fs, numtaps=120)

        if downsample_factor > 1:
            taps_antialias = self.fir_bandpass(1, fs_downsample / 2, fs, numtaps=120)

        # Initialize the feature array
        feature = np.zeros((data.shape[0], data.shape[1], len(time_downsampled)))
        bandpassed_signal = np.zeros((data.shape[0], data.shape[1], len(self.time)))

        for i in tqdm(range(data.shape[0])):
            # Filter the data
            signal_filtered = filtfilt(taps, [1.0], data[i, :, :], axis=1)

            # Apply Hilbert transform and get the envelope
            if enable_hilbert is True:
                signal_hilbert = np.abs(hilbert(signal_filtered, axis=1))
            else:
                signal_hilbert = signal_filtered

            if downsample_factor > 1:
                # Anti-aliasing filter if necessary
                signal_hilbert = filtfilt(taps_antialias, [1.0], signal_hilbert, axis=1)

                # Downsample the signal
                signal_downsampled = signal_hilbert[:, ::downsample_factor]
            else:
                signal_downsampled = signal_hilbert
            # Normalize if flag is set
            if normalize_signal:
                signal_downsampled, min_val, max_val = self.normalize_data(signal=signal_downsampled,
                                                                           moving_avg_size=moving_avg_size)
                signal_downsampled = 2 * signal_downsampled - 1
                signal_filtered = (signal_filtered - min_val) / (max_val - min_val)
                signal_filtered = 2 * signal_filtered - 1

            feature[i, :, :] = signal_downsampled
            bandpassed_signal[i, :, :] = signal_filtered
        return time_downsampled, feature, self.time, bandpassed_signal

    def get_all_bands_features(self, normalize_signal=False, moving_avg_size=0, sampling_freq=1000, feature_path=''):
        # Define the bands
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 45),
            "ripples": (80, 250),
            "fast_ripples": (250, 450)  # Assuming 500Hz as an upper limit for fast ripples
        }

        features = {}
        for band_name, (lowcut, highcut) in bands.items():
            features[band_name] = self.get_band_feature(band_name, lowcut, highcut,
                                                        feature_path=feature_path,
                                                        normalize_signal=normalize_signal,
                                                        moving_avg_size=moving_avg_size,
                                                        sampling_freq=sampling_freq)
        plot_eeg_bands(self.data, features, bands, idx=0, ch=0, t1=-0.5, t2=1.5, save_path=feature_path)

        return features

    def get_band_feature(self, band_name, lowcut, highcut, feature_path='', sampling_freq=250, normalize_signal=False,
                         moving_avg_size=0, enable_hilbert=True):
        norm_str = 'normalized' if normalize_signal else 'not_normalized'
        ma_str = f"ma_{moving_avg_size}" if moving_avg_size else 'no_ma'
        file_name = f"{band_name}_{norm_str}_{ma_str}_{sampling_freq}_features.pkl"
        file_path = feature_path + '_' + file_name
        Path(os.path.dirname(feature_path)).mkdir(parents=True, exist_ok=True)

        # Check if the feature file exists
        if os.path.isfile(file_path):
            print(f"Loading features for {band_name} from file.")
            with open(file_path, 'rb') as file:
                features = pickle.load(file)
        else:
            print(f"Extracting and saving features for {band_name}.")
            features = self.get_trial_features(
                self.data, lowcut, highcut, fs=self.fs, fs_downsample=sampling_freq,
                normalize_signal=False, moving_avg_size=0, enable_hilbert=enable_hilbert
            )
            # Save the extracted features
            with open(file_path, 'wb') as file:
                pickle.dump(features, file)

        return features

    def butter_bandpass(self, lowcut, highcut, order=4):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def extreact_continuous_data_from_epochs(self):
        num_trial, num_channel, num_sample = self.data.shape
        if self.trial_info is None:
            raise ValueError("To make continuous data from epoched data you need trial_info but it's None")

        trial_list = []
        event_indicator_list = []
        for i in self.trial_info['Trial number'].to_list():
            row = self.trial_info.loc[self.trial_info['Trial number'] == i, :]
            idx = row.index[0]
            # get the time interval from fixation to next fixateio
            time_range = [-row['fixation_time'].values[0], row['display_time'].values[0]]

            # find time idx
            t_start = np.argmin(np.abs(self.time - time_range[0]))
            t_end = np.argmin(np.abs(self.time - time_range[1]))
            t_origin = np.argmin(np.abs(self.time - 0))

            # get label from data info
            label = row['ColorLev'].values[0]

            # crate indicator signal
            event_indicator = np.zeros(self.time.shape)
            event_indicator[int(len(self.time) / 2):] = 1 if label == 1 else -1

            # truncated time
            trial_list.append(self.data[idx, :, t_start:t_end])
            event_indicator_list.append(event_indicator[t_start:t_end])

        continuous_data = np.concatenate(trial_list, axis=1)
        event_indicator = np.concatenate(event_indicator_list, axis=0)
        time = np.arange(0, len(event_indicator)) / self.fs

        return time, continuous_data, event_indicator

    def get_epochs_from_continuous_data(self, continuous_data):
        num_trial, num_channel, num_sample = self.data.shape
        epoched_data = np.zeros_like(self.data)
        if self.trial_info is None:
            raise ValueError("To make continuous data from epoched data you need trial_info but it's None")

        trial_list = []
        event_indicator_list = []
        for i in range(1, num_trial + 1):
            row = self.trial_info.loc[self.trial_info['Trial number'] == i, :]
            idx = row.index[0]
            # get the time interval from fixation to next fixation
            time_range = [-row['fixation_time'].values[0], row['display_time'].values[0]]

            # find time idx
            t_start = np.argmin(np.abs(self.time - time_range[0]))
            t_end = np.argmin(np.abs(self.time - time_range[1]))
            t_origin = np.argmin(np.abs(self.time - 0))

            epoched_data

            # get label from data info
            label = row['ColorLev'].values[0]

            # crate indicator signal
            event_indicator = np.zeros(self.time.shape)
            event_indicator[5000:] = 1 if label == 1 else -1

            # truncated time
            trial_list.append(self.data[idx, :, t_start:t_end])
            event_indicator_list.append(event_indicator[t_start:t_end])

        continuous_data = np.concatenate(trial_list, axis=1)
        event_indicator = np.concatenate(event_indicator_list, axis=0)
        time = np.arange(0, len(event_indicator)) / self.fs
