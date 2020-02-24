""" Utils to handle wav files
"""

import os
import numpy as np
import soundfile as sf


class SingleWav(object):
    """ Interface to handle a single wav file

    Args:
        file_name (str): The path to the wav file
        channel_interest (list[int]): An array of interested channels.
            Used in case of multichannel signals
        wav_id: An id to identify the wav file
        save (bool): Save the data untill the object is destroyed if True

    Examples:
        >>> SingleWav('/home/test.wav')

    """

    def __init__(self, file_name, channel_interest=None, wav_id=None,
                 save=False):
        self.file_name = file_name
        self.__wav_data = None
        self.__id = wav_id
        self.info = None
        self.sampling_rate = None
        self.sample_len = None
        self.channel_count = None
        self.save = save
        self.channel_interest = None
        self.verify()
        if channel_interest is not None:
            self.channel_interest = np.array(channel_interest)

    def verify(self):
        """ Verify if all the information is good """
        assert os.path.exists(self.file_name), (self.file_name +
                                                ' does not exists')

    def update_info(self):
        """ Get wav related info and place it in the :attr:`info` variable.

        .. note:: Avoid calling this in the `__init__` section. Very time
            consuming
        """
        if self.info is None:
            self.info = sf.info(self.file_name)
            self.sampling_rate = self.info.samplerate
            self.sample_len = int(self.info.samplerate * self.info.duration)
            self.channel_count = self.info.channels

    @property
    def wav_len(self):
        """ Get the sample length of the signal

        Returns:
            int: Wav length in number of samples
        """
        if self.sample_len is None:
            self.update_info()
        return self.sample_len

    @property
    def data(self):
        """ Read the wav file if not saved

        Returns:
           :class:`numpy.ndarray`:
                Two dimensional array of shape [samples, channels]
        """
        self.update_info()
        if self.__wav_data is not None:
            return self.__wav_data
        wav_data, self.sampling_rate = sf.read(self.file_name, always_2d=True)
        if self.channel_interest is not None:
            wav_data = wav_data[:, self.channel_interest]
        if self.save:
            self.__wav_data = wav_data
        return wav_data

    def random_part_data(self, duration=-1):
        """
            Return random part of the wav file
            Args:
                duration: float. required duration in seconds.
                    defaults to -1 in which case returns the full signal

            Returns:
                A two dimensional numpy array of shape [samples, channels]
        """
        if duration == -1:
            return self.data
        self.update_info()
        assert duration < self.info.duration,\
                'Requested duration exceeds signal length'
        max_sample = int((self.info.duration - duration) * self.sampling_rate)
        start_sample = np.random.randint(0,max_sample)
        end_sample = start_sample + int(duration * self.sampling_rate)
        return self.part_data(start_sample, end_sample)

    def part_data(self, start, end):
        """
            Read part of the wav file
        Args:
            start: int, start of the wav file (in samples)
            end: int, end of the wav file in samples
        Returns
            A two dimensional numpy array of shape [samples, channels]
        """
        self.update_info()
        assert end > start, 'End should be greater than start'
        assert end <= self.sample_len,\
                'Requested length is greater than max available'
        if self.__wav_data is None:
            wav_data, self.sampling_rate = \
                    sf.read(self.file_name, always_2d=True, 
                            start=start, stop=end, dtype='float32')
        else:
            wav_data = self.__wav_data[start:end, :]
        if self.channel_interest is not None:
            wav_data = wav_data[:, self.channel_interest]
        return wav_data

    def __enter__(self):
        """ Using `with` operator for wav object
        Examples:
            >>> from asteroid.data.wav import SingleWav
            >>> from asteroid.filterbanks import Encoder, STFTFB
            >>> wav_obj = SingleWav('file.wav')
            >>> fb = Encoder(STFTFB(512, 512))
            >>> set_trace()
            >>> with wav_obj: # Wav file is read
            >>>     print(wav_obj._SingleWav__wav_data is None)
            False
            >>>     data = torch.tensor(wav_obj.data,
                        dtype=torch.float32).T.unsqueeze(1)
            >>>     data_stft = fb(data)
            ## Picks wav data from memory and not from file
            >>>     new_data = torch.tensor(wav_obj.data.sum(1),
                        dtype=torch.float32).T.unsqueeze(0).unsqueeze(0)
            >>>     new_data_stft = fb(new_data)
            >>> # Wav data cleared from memory
            >>> print(wav_obj._SingleWav__wav_data is None)
            True
        """
        self.temp_save()

    def __exit__(self, data_type, data_val, data_tb):
        """ Clear wav data is save not requested
        """
        if not self.save:
            self.save_space()

    def save_space(self):
        """ Remove the saved data. self.data will read from the file again.
        """
        self.__wav_data = None

    def temp_save(self):
        """ Temporarily save the wav data.

        Call :func:`save_space` to remove it.
        """
        self.__wav_data = self.data

    @property
    def wav_id(self):
        """ Get wav id """
        return self.__id

    @wav_id.setter
    def wav_id(self, value):
        self.__id = value

    def write_wav(self, path):
        """ Write the wav data into an other path """
        sf.write(path, self.data, self.sampling_rate)


class MultipleWav(SingleWav):
    """ Handle a set of wav files as a single object.
    Args:
        file_name_list (list[str]): List of wav file names
        channel_interest (list[int]): An array of interested channels.
            Used in case of multichannel signals
        wav_id: An id to identify the bunch of wav file
        save (bool): Save the data until the object is destroyed if True

    """

    def __init__(self, file_name_list, channel_interest=None, wav_id=None,
                 save=False):
        self.file_name_list = file_name_list
        self.__wav_data = None
        self.__id = wav_id
        self.info = None
        self.sampling_rate = None
        self.sample_len = None
        self.channel_count = None
        self.info_list = []
        self.sampling_rate_list = []
        self.sample_len_list = []
        self.channel_count_list = []
        self.save = save
        self.channel_interest = None
        if channel_interest is not None:
            self.channel_interest = np.array(channel_interest)

    def update_info(self):
        if self.info is None:
            for _file_ in self.file_name_list:
                info = sf.info(_file_)
                self.info_list.append(info)
                self.sampling_rate_list.append(info.samplerate)
                self.sample_len_list.append(
                    int(info.samplerate * info.duration))
                self.channel_count_list.append(info.channels)
            self.info = info
            self.sampling_rate = info.samplerate
            self.sample_len = int(info.samplerate * info.duration)
            self.channel_count = info.channels

    @property
    def data(self):
        """ Reads all the files in the file list

        Returns:
            list[:class:`numpy.ndarray`]:
                A list of wav signals
        """
        self.update_info()
        if self.__wav_data is not None:
            return self.__wav_data
        wav_data = []
        for _file_ in self.file_name_list:
            _wav_data, _ = sf.read(_file_, always_2d=True, dtype='float32')
            if self.channel_interest is not None:
                _wav_data = _wav_data[:, self.channel_interest]
            wav_data.append(_wav_data)
        if self.save:
            self.__wav_data = wav_data
        return wav_data
