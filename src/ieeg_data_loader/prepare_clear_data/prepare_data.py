import json
import pandas as pd
import h5py
import numpy as np
import os
import warnings
import ast
import pickle
from pathlib import Path


class CLEARDataLoader:
    def __init__(self, base_path='D:/Navid/Dataset/CLEAR/raw/', save_base_path='D:/Navid/Dataset/CLEAR/interim/'):
        self.patient_names = ['p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10',
                              'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18']
        self.task_names = ['flicker', 'm_sequence', 'imagine', 'flicker_shape']
        self.task_names_dict = {1: 'flicker', 2: 'm_sequence', 3: 'imagine', 4: 'flicker_shape'}
        self.base_path = base_path
        self.save_base_path = save_base_path

    def prepare_data(self, patient_names=None):
        """

        :param patient_names:
        :return:
        """
        data_info = pd.read_csv("setting_files/Dataset_summary.csv")
        if patient_names is None:
            patient_names = self.patient_names
        for index, row in data_info.iterrows():
            print(
                f"{index} from {data_info.shape[0]}: \tpatient: {row['patient']} "
                f"\ttask: {self.task_names_dict[int(row['trial_type'])][:13]:13s} "
                f"\tblock: {int(row['block_number'])} \ttrials: {int(row['number of trials'])}")
            file_name = f"{row['patient']}_task{int(row['trial_type'])}_block{int(row['block_number'])}_" \
                        f"{row['directory'].split('/')[-2]}"
            save_path = self.save_base_path + '/{}/{}/'.format(row['task'], row['patient'])
            if os.path.exists(save_path + file_name + '.pkl'):
                print("file exists")
            else:
                merged_bank_file, field_trip_file = self.load_data_row(row)
                idx_row, trial_info_row, data_row = self.get_trial_info_and_data_row(row, merged_bank_file,
                                                                                     field_trip_file)
                data_dict = self.convert_fieldtrip_to_dict(row, merged_bank_file, field_trip_file, idx_row,
                                                           trial_info_row,
                                                           data_row)
                data_dict, label = self.get_labels(data_dict)
                self.save_data(data_dict, file_name, save_path)

    def load_data_row(self, row):
        """
        load raw data from .mat files
        :param row:
        :return:
        """
        main_direction, file1, file2 = row['directory'], row['file1'], row['file2']
        path_dataset_fieldtrip = self.base_path + main_direction + file1
        path_dataset_mergedbank = self.base_path + main_direction + file2

        merged_bank_file = h5py.File(path_dataset_mergedbank, 'r')
        field_trip_file = h5py.File(path_dataset_fieldtrip, 'r')

        return merged_bank_file, field_trip_file

    @staticmethod
    def get_trial_info_and_data_row(row, merged_bank_file, field_trip_file):
        """
        Gets the files and a row from dataset csv info and get trial info
        :param row:
        :param merged_bank_file:
        :param field_trip_file:
        :return:
        """
        # get row information
        header = ast.literal_eval(row['header'])
        block_number_index = header.index('BlockNumber')

        row_experiment_type, row_block_number = row['trial_type'], row['block_number']

        # load data
        data_list = [field_trip_file[ref[0]][()].transpose() for ref in field_trip_file['ft_data3']['trial'][()]]
        data = np.stack(data_list, axis=0)

        # load trial info
        experiment_type = np.squeeze(merged_bank_file['TrialTypeDesignation'][()])
        if any(np.isnan(experiment_type)):
            experiment_type = row_experiment_type * np.ones_like(experiment_type)
            warnings.warn("TrialTypeDesignation contains NaN values!")

        num_trial = row['number of trials']
        trial_info = field_trip_file[row['trial_info_tag']][()].transpose()
        if data.shape[0] != trial_info.shape[0]:
            warnings.warn("Data shape is {} but trial info shape is {}".format(data.shape, trial_info.shape))

        # find trial ino
        idx_experiment_type = np.where(experiment_type == row_experiment_type)[0]
        if trial_info.shape[0] != len(experiment_type):
            warnings.warn(
                "TrialTypeDesignation shape is {} but trial info shape is {}".format(len(experiment_type), trial_info.shape))
            if data.shape[0] == trial_info.shape[0]:
                idx_experiment_type = np.array(range(trial_info.shape[0]))
                experiment_type = row_experiment_type * np.ones_like(experiment_type)
            else:
                idx_experiment_type = idx_experiment_type[:trial_info.shape[0]]

        trial_info_df = pd.DataFrame(trial_info[idx_experiment_type], columns=header)
        trial_info_df['TrialTypeDesignation'] = experiment_type[idx_experiment_type]

        if 'Onset Choice Period' in trial_info_df.columns.to_list() and len(trial_info_df['Onset Choice Period'].unique()) == 1:
            trial_info_df['Onset Choice Period'] = trial_info_df['CrossStart'] - trial_info_df['StartTrial']

        trial_info_row = trial_info_df[trial_info_df['BlockNumber'] == row_block_number]
        indices = trial_info_row.index
        idx_row = idx_experiment_type[indices]

        # check the dimensions
        assert num_trial == len(idx_experiment_type[indices])
        assert num_trial == trial_info_row.shape[0]
        assert all(trial_info[idx_row, block_number_index] == row_block_number)

        return idx_row, trial_info_row.reset_index(drop=True), data[idx_row]

    @staticmethod
    def convert_fieldtrip_to_dict(row, merged_bank_file, field_trip_file, idx_row, trial_info_row, data_row):
        """

        :param row:
        :param merged_bank_file:
        :param field_trip_file:
        :param idx_row:
        :param trial_info_row:
        :param data_row:
        :return:
        """
        ft_data = field_trip_file['ft_data3']
        time = np.squeeze(field_trip_file[field_trip_file['ft_data3']['time'][()][0][0]][()])

        if len(merged_bank_file['ChannelPairNamesBank1'].shape) > 1:
            channel_names = ["".join(chr(char) for char in cell) for cell in
                             merged_bank_file['ChannelPairNamesBank1'][()].transpose()]
            if len(merged_bank_file['ChannelPairNamesBank2'].shape) > 1:
                channel_names = channel_names + ["".join(chr(char) for char in cell) for cell in
                                                 merged_bank_file['ChannelPairNamesBank2'][()].transpose()]
        elif len(merged_bank_file['ChannelPairNames'].shape) > 1:
            channel_names = ["".join(chr(char) for char in cell) for cell in
                             merged_bank_file['ChannelPairNames'][()].transpose()]
        else:
            channel_names = [str(ch) for ch in range(data_row.shape[1])]
            warnings.warn(" No Channel Name is available")

        if len(channel_names) != data_row.shape[1]:
            warnings.warn("Number of channels are {} but data dimension is {}".format(len(channel_names), data_row.shape))

        data_dict = {
            "data": data_row,
            "time": time,
            "channel_names": channel_names,
            "trial_info": trial_info_row,
            "fs": np.squeeze(ft_data['fsample'][()]),
            "index": idx_row
        }
        data_dict.update(row)

        return data_dict

    def save_data(self, data_dict, file_name, save_path=None):
        if save_path is None:
            save_path = self.save_base_path + '/{}/{}/'.format(data_dict['task'], data_dict['patient'])
        Path(save_path).mkdir(parents=True, exist_ok=True)

        with open(save_path + file_name + '.pkl', 'wb') as f:
            pickle.dump(data_dict, f)

        data_dict['trial_info'].to_csv(save_path + file_name + '.csv')

        print(f"{data_dict['data'].shape} trials saved successfully")

    def get_labels(self, data_dict):
        trial_info = data_dict['trial_info']
        if data_dict['trial_type'] == 1:
            label = trial_info['ColorLev']
        elif data_dict['trial_type'] == 2:
            label_m_seq_data = h5py.File(self.base_path + 'SequenceDisplay.mat', 'r')
            label_m_seq = label_m_seq_data['seq'][()]
            delay1, delay2 = label_m_seq_data['TrialDelay1'][()], label_m_seq_data['TrialDelay2'][()]
            ms = label_m_seq_data['ms'][()]
            label = label_m_seq
            if label.shape[0] != data_dict['data'].shape[0]:
                warnings.warn("M-sequence has {} label but we have 100 sequence label".format(
                    data_dict['data'].shape[0]))
        elif data_dict['trial_type'] == 3:
            label = trial_info['ColorLev']
        elif data_dict['trial_type'] == 4:
            label = trial_info[['ColorLev', 'ShapeChoice']]
        data_dict.update({'label': label})
        return data_dict, label

    def create_data_report_csv(self):
        raw_dataset = 'D:/Navid/Dataset/CLEAR/'
        # print_directory_structure(raw_dataset)
        data_rows = []
        for patient in self.patient_names:
            print("\n\n=================Patient {}=====================".format(patient))

            path_dataset_fieldtrip = []
            path_dataset_mergedbank = []
            for task in ['flicker_task', 'imagine_task', 'flicker_shape_task']:
                with open('../DataPaths/{}.json'.format(task), 'r') as f:
                    dataset_path_dict = json.load(f)
                print("\npatient {} \t task = {}".format(patient, task))
                for i in range(len(dataset_path_dict[patient])):
                    main_direction, file1, file2 = dataset_path_dict[patient][i]
                    formatted_main_direction = (main_direction[:20] + ' ' * 20)[:20]
                    print("{}  \t file1: {}  \t file2:{}".format(formatted_main_direction, file1, file2))
                    path_dataset_fieldtrip.append(raw_dataset + main_direction + file1)
                    path_dataset_mergedbank.append(raw_dataset + main_direction + file2)

                    data, trial_info, num_channel, trial_info_columns, trial_info_tag = \
                        self.get_trial_info(patient=patient,
                                            path_dataset_fieldtrip=path_dataset_fieldtrip[-1],
                                            path_dataset_mergedbank=path_dataset_mergedbank[-1],
                                            task=task[:-5],
                                            file_idx=i + 1)
                    if data.shape[0] != trial_info.shape[0]:
                        print("unmatched data")
                    grouped = trial_info.groupby('TrialTypeDesignation')['BlockNumber'].value_counts()
                    print("Num channel {}".format(num_channel))
                    print("{} {}\t  --- trial types {}".format(task, i + 1,
                                                               trial_info[
                                                                   'TrialTypeDesignation'].value_counts().to_json()))
                    for unique_value in trial_info['TrialTypeDesignation'].unique():
                        filtered_trial_info = trial_info[trial_info['TrialTypeDesignation'] == unique_value]
                        block_info = filtered_trial_info['BlockNumber'].value_counts().to_json()
                        print("\t Blocks For TrialType {}: \t {}".format(unique_value, block_info))

                        row = {
                            'patient': patient,
                            'task': task,
                            'directory': main_direction,
                            'num_channel': num_channel,
                            'file1': file1,
                            'file2': file2,
                            'trial_type': unique_value,
                            'block_info': block_info,
                            'header': trial_info_columns,
                            'trial_info_tag': trial_info_tag,
                        }
                        # Add the new row to your list
                        data_rows.append(row)
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        df.to_csv("Dataset_summary.csv", index=False)

    def get_trial_info(self, patient, path_dataset_mergedbank, path_dataset_fieldtrip, task, file_idx):
        merged_bank_file = h5py.File(path_dataset_mergedbank, 'r')
        field_trip_file = h5py.File(path_dataset_fieldtrip, 'r')

        data_list = [field_trip_file[ref[0]][()].transpose() for ref in field_trip_file['ft_data3']['trial'][()]]
        data = np.stack(data_list, axis=0)

        experiment_type = np.squeeze(merged_bank_file['TrialTypeDesignation'][()])
        if file_idx == 1 and patient == 'p04':
            print("mismatch between TrialTypeDesignation ({}) and data size ({}) \n".format(len(experiment_type),
                                                                                            data.shape[0]))
            experiment_type = experiment_type[:data.shape[0]]

        if data.shape[0] != len(experiment_type):
            print("mismatch between TrialTypeDesignation ({}) and data size ({}) \n".format(len(experiment_type),
                                                                                            data.shape[0]))
            experiment_type = np.ones_like(experiment_type[:data.shape[0]])
            if '_IMAGINE_' in path_dataset_fieldtrip:
                experiment_type = experiment_type * 3
            elif '_MSEQUENCE_' in path_dataset_fieldtrip:
                experiment_type = experiment_type * 2

        if patient.lower() == 'p13' and any(np.isnan(experiment_type)):
            experiment_type[np.isnan(experiment_type)] = 3
        trial_info_tag = 'TrialAlignedMat1'
        if '_IMAGE_' in path_dataset_fieldtrip:
            trial_info_tag = 'TrialAlignedMat1'
            trial_info_hdr = [np.squeeze(field_trip_file[ref[0]][()]) for ref in
                              field_trip_file['TrialAlignedMatHeaders'][()]]
            trial_info_hdr = ["".join(chr(char) for char in list(info)) for info in trial_info_hdr]
            trial_info = field_trip_file[trial_info_tag][()].transpose()

            if trial_info.shape[-1] == len(trial_info_hdr):
                columns_name = trial_info_hdr
            elif patient in ['p05']:
                columns_name = ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'Stim on/off NEV',
                                'Stim onset NEV', 'Stim offset NEV', 'Response value NEV', 'Reaction onset NEV',
                                'Congruent Value NEV', 'Valence Value NEV', 'Reaction time NEV', 'Noname', 'TrialNum',
                                'Delay1', 'Delay2', 'TrialComplete', 'TimeImageOnset', 'ColorLev', 'TimeRec',
                                'TimeTrial']
                if file_idx == 2:
                    columns_name.extend(['noname1', 'noname2', 'noname3'])
                columns_name.append('BlockNumber')
            elif patient in ['p06']:
                columns_name = ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'Stim on/off NEV',
                                'Stim onset NEV', 'Stim offset NEV', 'Response value NEV', 'Reaction onset NEV',
                                'Congruent Value NEV', 'Valence Value NEV', 'Reaction time NEV', 'Noname', 'TrialNum',
                                'Delay1', 'Delay2', 'TrialComplete', 'TimeImageOnset', 'ColorLev', 'TimeRec',
                                'TimeTrial',
                                'noname1', 'noname2', 'noname3', 'BlockNumber']
            elif patient in ['p01', 'p03', 'p04']:
                columns_name = ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'Stim on/off NEV',
                                'Stim onset NEV', 'Stim offset NEV', 'Response value NEV', 'Reaction onset NEV',
                                'Congruent Value NEV', 'Valence Value NEV', 'Reaction time NEV', 'TrialNum',
                                'Delay1', 'Delay2', 'TrialComplete', 'TimeImageOnset', 'ColorLev',
                                'TimeRec', 'TimeTrial', 'BlockNumber']
            elif patient in ['p02']:
                columns_name = ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'Stim on/off NEV',
                                'Stim onset NEV', 'Stim offset NEV', 'Response value NEV', 'Reaction onset NEV',
                                'Congruent Value NEV', 'Valence Value NEV', 'Reaction time NEV', 'TrialNum',
                                'Delay1', 'Delay2', 'TrialComplete', 'TimeImageOnset', 'Noname', 'ColorLev',
                                'TimeTrial', 'BlockNumber']
            elif patient in ['p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18']:
                columns_name = ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'NoName', 'Stim on/off NEV',
                                'Stim onset NEV', 'Stim offset NEV',  # 'Response value NEV', 'Reaction onset NEV'
                                'Congruent Value NEV', 'Valence Value NEV', 'Reaction time NEV', 'NoName1', 'NoName2',
                                'TrialNum',
                                'Delay1', 'Delay2', 'TrialComplete', 'TimeImageOnset', 'ColorLev',
                                'TimeRec', 'TimeTrial', 'StartTrial', 'CrossStart', 'ResponseTime']
                if 'ShapeChoice' in trial_info_hdr:
                    columns_name.append('ShapeChoice')
                columns_name.append('BlockNumber')
            else:
                raise ValueError("No Trial infor alignment is specified for {}".format(patient))
        elif '_IMAGINE_' in path_dataset_fieldtrip:
            trial_info_tag = 'TrialAlignedMat1Img'
            trial_info_hdr = [np.squeeze(field_trip_file[ref[0]][()]) for ref in
                              field_trip_file['TrialAlignedMatHeadersImagine'][()]]
            trial_info_hdr = ["".join(chr(char) for char in list(info)) for info in trial_info_hdr]
            trial_info = field_trip_file[trial_info_tag][()].transpose()
            if trial_info.shape[-1] == len(trial_info_hdr):
                columns_name = trial_info_hdr
            elif patient == 'p05':
                columns_name = ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'Image Offset NEV', 'nothing',
                                'Fixation Onset NEV', 'Onset Choice Period', 'Response time NEV', 'Shade value NEV',
                                'Reaction time NEV', 'Trial onset', 'Trial offset', 'nothing', 'TrialNum', 'Delay1',
                                'Delay2',
                                'TrialComplete', 'TimeImageOnset', 'ColorLev', 'TimeRec', 'TimeTrial', 'StartTrial',
                                'CrossStart', 'ResponseTime', 'BlockNumber', 'TrialTypeDesignation']
            else:
                columns_name = trial_info_hdr
                if 'BlockNumber' not in columns_name:
                    columns_name.append('BlockNumber')
                # raise ValueError("Not implemented")
        elif '_MSEQUENCE_' in path_dataset_fieldtrip:
            trial_info_tag = 'TrialAlignedMat1Mseq'
            trial_info_hdr = [np.squeeze(field_trip_file[ref[0]][()]) for ref in
                              field_trip_file['TrialAlignedMatHeadersMseq'][()]]
            trial_info_hdr = ["".join(chr(char) for char in list(info)) for info in trial_info_hdr]
            trial_info = field_trip_file[trial_info_tag][()].transpose()
            print(
                "Trial info shape {} \t data shape {} \t TrialTypeDesignation shapr {}".format(trial_info.shape,
                                                                                               data.shape,
                                                                                               experiment_type.shape))
            if trial_info.shape[-1] == len(trial_info_hdr):
                columns_name = trial_info_hdr
            else:
                columns_name = trial_info_hdr
                if 'BlockNumber' not in columns_name:
                    columns_name.append('BlockNumber')
        else:
            raise ValueError("Task not defined")
        if trial_info.shape[1] > len(columns_name):
            columns_name = columns_name + ['noname'] * (trial_info.shape[1] - len(columns_name))
        else:
            columns_name = columns_name[:trial_info.shape[1]]
        trial_info = pd.DataFrame(trial_info, columns=columns_name)
        trial_info['TrialTypeDesignation'] = experiment_type
        trial_info.to_csv('{}_{}_{}.csv'.format(patient, file_idx, task), index=False)
        return data, trial_info, data.shape[1], columns_name, trial_info_tag

    @staticmethod
    def print_directory_structure(startpath):
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))
