# Imports.
import numpy as np
import soundfile as sf
import audiosegment
import matplotlib.pyplot as plt

from yamnet import params as yamnet_params
from yamnet import yamnet as yamnet_model
import tensorflow as tf
import IPython

class YamnetEval:
    def load(self):
        params = yamnet_params.Params(sample_rate=44100, patch_hop_seconds=0.1)
        # print("Sample rate =", params.sample_rate)
        self.class_names = yamnet_model.class_names('yamnet/yamnet_class_map.csv')
        self.yamnet = yamnet_model.yamnet_frames_model(params)
        self.yamnet.load_weights('yamnet/yamnet.h5')

    def process(self,file_path,debug=[]):
        audio = audiosegment.from_file(file_path)
        IPython.display.display(audio)
        wav_data=audio.to_numpy_array()
        if(audio.channels>1):
            wav_data=wav_data[:,0]
        sr=audio.frame_rate
        if(sr!=44100):
            print(f'Errorrrrrrrrrrrr framerate={sr}')
        waveform = wav_data / 32768.0
        wav_dur=audio.duration_seconds
        # print("Information:")
        # print(f'sample_rate: {audio.frame_rate} duration: {audio.duration_seconds}s')
        # print("Channels:", audio.channels)
        # print("Bits per sample:", audio.sample_width * 8)
        # print("Sampling frequency:", audio.frame_rate)
        # print("Length:", audio.duration_seconds, "seconds")
        # The graph is designed for a sampling rate of 16 kHz, but higher rates should work too.
    # We also generate scores at a 10 Hz frame rate.
        
        scores, embeddings, spectrogram = self.yamnet(waveform)
        scores = scores.numpy()
        spectrogram = spectrogram.numpy()
        
        if 'V' in debug:
            self.draw(scores,spectrogram,waveform,wav_dur)
        return self.convert2Events(scores,wav_dur)

    def draw(self,scores,spectrogram,waveform,wav_dur):
        # Visualize the results.
        plt.figure(figsize=(10, 8))
        times = np.linspace(0, len(waveform) / 44100, num=len(waveform))
        # Plot the waveform.
        plt.subplot(3, 1, 1)
        plt.plot(times,waveform)
        plt.xlim([0, wav_dur])
        # Plot the log-mel spectrogram (returned by the model).
        plt.subplot(3, 1, 2)
        plt.imshow(spectrogram.T, aspect='auto', interpolation='nearest', origin='upper',extent=[0,wav_dur,65,0])

        # Plot and label the model output scores for the top-scoring classes.
        mean_scores = np.mean(scores, axis=0)
        top_N = 10
        top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
        top_class_indices = [294,421]
        top_N=len(top_class_indices)
        plt.subplot(3, 1, 3)
        plt.imshow(scores[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r',extent=[0,wav_dur,top_N-.5,0-.5])
        # Compensate for the patch_window_seconds (0.96s) context window to align with spectrogram.
        # patch_padding = (params.patch_window_seconds / 2) / params.patch_hop_seconds
        # plt.xlim([-patch_padding, scores.shape[0] + patch_padding])
        # Label the top_N classes.
        yticks = range(0, top_N, 1)
        plt.yticks(yticks, [self.class_names[top_class_indices[x]] for x in yticks])
        _ = plt.ylim(-0.5 + np.array([top_N, 0]))

    def convert2Events(self,scores,wav_dur):
        threshold=.5
        top_N = 4
        mean_scores = np.mean(scores, axis=0)
        top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
        top_class_indices = [294,421]
        events_dict={self.class_names[c]:[] for c in top_class_indices}
        events=[]
        rate=wav_dur/scores.shape[0]
        tmp={c:{'s':-1, 'p':0} for c in top_class_indices}
        for i in range(scores.shape[0]):
            for x in top_class_indices:
                p=scores[i][x]
                if(p>=threshold and tmp[x]['s']==-1):
                    tmp[x]['s']=i
                    tmp[x]['p']=p
                elif (scores[i][x]<threshold and tmp[x]['s']!=-1):
                    s=tmp[x]['s']
                    newE=(s*rate,(i)*rate,tmp[x]['p']/(i-s))
                    events_dict[self.class_names[x]].append(newE)
                    tmp[x]['s']=-1
                else:
                    tmp[x]['p']+=p

        for c in events_dict:
            events.extend([(k[0],k[1],c,k[2]) for k in events_dict[c]])
        import pandas as pd
        eventdf=pd.DataFrame(events,columns=['start_time_seconds','end_time_seconds','label_full','p'])

        return eventdf
