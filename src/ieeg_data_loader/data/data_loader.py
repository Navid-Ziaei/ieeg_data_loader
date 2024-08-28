from ..data.dataset import IEEGData
from abc import ABC, abstractmethod
from scipy import io
import os
import numpy as np
import pandas as pd


class AbstractDataLoader(ABC):
    def __init__(self, patient, target_class, task, prepared_dataset_path, file_format):
        self.target_class = target_class
        self.patient = patient
        self.task = task  # 'FlickerShapes' or 'Flicker'
        self.prepared_dataset_path = prepared_dataset_path
        self.file_format = file_format
        self.results = {}

    @abstractmethod
    def read_data(self, patient_dir_path, file_dir):
        pass

    def load_data(self):
        data_list = []
        if isinstance(self.task, str):
            task = [self.task]
        else:
            task = self.task
        if isinstance(self.patient, str):
            patient = [self.patient]
        else:
            patient = self.patient

        file_dir = [[], [], []]
        for task_dir in os.listdir(self.prepared_dataset_path):
            if task and task_dir.replace('_task', '') not in task:
                continue
            file_dir[0] = task_dir
            task_dir_path = os.path.join(self.prepared_dataset_path, task_dir)
            if os.path.isdir(task_dir_path):
                for patient_dir in os.listdir(task_dir_path):
                    if patient and patient_dir not in patient:
                        continue
                    file_dir[1] = patient_dir
                    patient_dir_path = os.path.join(task_dir_path, patient_dir)
                    new_subject_data = self.read_data(patient_dir_path=patient_dir_path,
                                                      file_dir=file_dir)
                    data_list.extend(new_subject_data)
        if len(data_list) == 0:
            print(f" No data found for patient {self.patient} and task {self.task}")
        return data_list

    def get_available_blocks(self):
        pass


class iEEGDataLoader(AbstractDataLoader):
    def __init__(self, patient, target_class, prepared_dataset_path, task='flicker', file_format='npy'):
        super().__init__(patient, target_class, task, prepared_dataset_path, file_format)
        # Add any additional initialization if needed

    def read_data(self, patient_dir_path, file_dir):
        i = 0
        data_ieeg_list = []
        for file in os.listdir(patient_dir_path):
            if file.endswith('.pkl'):
                file_dir[2] = file
                file_path = os.path.join(patient_dir_path, file)
                data = pd.read_pickle(file_path)
                print(f"{i}: Data {file} \t Block Number: {data['trial_info']['BlockNumber'].unique()} \t"
                      f"Number of trials: {data['trial_info'].shape[0]}")
                if data['trial_info'].shape[0] > 100:
                    pass
                else:
                    # Change the format to dataframe
                    if len(data['label'].shape) > 2 and data['label'].shape[2] == 10:
                        label_matrix = data['label'].reshape(100, 100)
                        columns = [f"target_{i}_{j}" for i in range(10) for j in range(10)]
                        data['label'] = pd.DataFrame(label_matrix, columns=columns)
                    else:
                        data['label'] = pd.DataFrame(data['label'])

                    # Remove the Nan Trials
                    nan_trials = list(np.unique(np.where(np.isnan(data['data']))[0]))
                    if len(nan_trials) > 0:
                        print(f"Warning: Trial {nan_trials} contains NaN. It is removed")
                        data['data'] = np.delete(data['data'], nan_trials, axis=0)
                        data['trial_info'] = data['trial_info'].drop(nan_trials, axis=0)
                        data['index'] = np.delete(data['index'], nan_trials, axis=0)
                        data['label'] = data['label'].drop(nan_trials, axis=0)
                    data_ieeg = IEEGData()
                    data_ieeg.dict_to_iEEG_format(data_dict=data.copy(), meta=file_dir.copy())

                    i = i + 1
                    data_ieeg_list.append(data_ieeg)
        return data_ieeg_list


class OCEDEEGDataLoader(AbstractDataLoader):
    def __init__(self, paths, settings):
        super().__init__(paths, settings)
        # Add any additional initialization if needed

    def read_data(self, patient_dir_path, file_dir):
        i = 0
        for file in os.listdir(patient_dir_path):
            if file.endswith(self.file_format):
                file_dir[2] = file
                file_path = os.path.join(patient_dir_path, file)
                patient_data_mat = io.loadmat(file_path)
                # extract data from mat file
                fs = patient_data_mat['fs'][0][0]
                target_example = np.squeeze(patient_data_mat['exemplarLabels']) - 1
                target_category = np.squeeze(patient_data_mat['categoryLabels']) - 1
                data = patient_data_mat['xEpoched'][:-1]
                num_electrodes, num_samples, num_trials = data.shape
                data = np.transpose(data, (2, 0, 1))
                time = np.squeeze(patient_data_mat['epochTimeSamples'])
                unique_categories = np.unique(target_category) - 1
                onsets = np.squeeze(patient_data_mat['onsets'])
                trial_info = {'Trial number': list(np.arange(0, num_trials)),
                              'Block number': patient_data_mat['sessionID'][0],
                              'exemplar labels': target_example,
                              'category labels': target_category,
                              'onsets': onsets}

                print(f"{i}: Data {file} \t session ID: {patient_data_mat['sessionID'][0]} \t"
                      f"Number of trials: {data.shape[0]}")

                channel_name = list(np.arange(0, num_electrodes))
                data_ieeg = IEEGData()
                data_ieeg.data = data
                data_ieeg.channel_name = [str(ch) for ch in channel_name]
                data_ieeg.trial_info = pd.DataFrame(trial_info)
                data_ieeg.data_info = trial_info
                data_ieeg.time = time / 1000  # time should be in second
                data_ieeg.label = target_category
                data_ieeg.fs = fs
                data_ieeg.meta = file_dir

                i = i + 1
        return data_ieeg


