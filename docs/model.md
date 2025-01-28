# Cherry Lip Sync Model

The Cherry Lip Sync model was developed based on the ideas from
_Real-Time Lip Sync for Live 2D Animation_[^1].

## The Model

The model starts with audio mixed to mono and resampled to 16 kHz. The audio is
windowed into segments of length 25 ms spaced 10 ms apart using a [Hann
window](https://en.wikipedia.org/wiki/Hann_function). A Fourier transform is
done to each window then various sets of results are used to compute power in
[Mel bands](https://en.wikipedia.org/wiki/Mel_scale). This converts the
frequency space from linear to something more similar to human hearing. The Mel
spectrum is computed for 13 banks. Then derivatives of the Mel spectrum are
taken smoothing over two frames on each side to smooth it out. The final feature
vector input has 26 values for each frame.

The model starts with a layer of [batch
normalization](https://en.wikipedia.org/wiki/Batch_normalization) to learn
ranges of the input. Then the model is two layers of unidirectional
[GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) each with hidden state
size 80. After the second GRU there is a linear layer that transforms from size
80 to size 12. The output is interpreted as logit probabilities for the 12
possible visemes. For inference, the highest output value is chosen at each time
step. The output is interpreted for time 3 frames ahead. This means the model
"looks into the future" by 3 frames to predict the current output. This
lookahead helps the model to anticipate viseme changes that happen before audio
data is available. 

The output is available at 100 Hz (because of the window spacing). The command
line tool performs inference and then samples the model outputs at the final
desired framerate. The command line interface has an optional filtering step
that can prevent single frame visemes.

Some of the notable things NOT present:
1) There is no Mel spectrum normalization beyond batch norm
2) There is no MFCC computation
3) The model is unidirectional, does not need to know all future audio
4) The lookahead is limited to 3 frames (compared to 6 in reference[^1])
5) The model does not need to know text or phonemes in audio

## Visualization



## Training Data

The training set is a proprietary lip sync dataset generated from public domain
audio. The main idea was to use manually annotated data sources with the same
sentence said by many people. One manual lip sync sequence then can be used many
times for training with a variety of audio frames. Another idea was to warp
existing audio and lip sync timing data by time stretching and pitch correction.
Finally synthetic audio could be generated with precisely known phoneme timing
and mapped to plausible visemes, then manually reviewed quickly (e.g. 1 hour of
review to generate 10 minutes of usable training data).

The current training dataset contains about 1 hour of audio and takes about 5
minutes to fully train with 200 epochs on an NVIDIA GTX 3090.

## Python

The model was developed and trained with [PyTorch](https://pytorch.org/). The
`training/` subdirectory contains the model definition, data pipeline, scripts,
and some disorganized Python notebooks. There is also an inference test using
random input. The model and various vectors are saved in `pt` format.

## Rust

The main project is written in Rust using the [`burn` deep learning
framework](https://burn.dev/). There is a separate binary that reads the `pt`
model and vectors and converts them to Rust `bin` format.

The main command line tool includes the `bin` format model inside the executable
to avoid needing any configuration files or setup.


[^1]: D. Aneja, W. Li. _Real-Time Lip Sync for Live 2D Animation_. https://arxiv.org/abs/1910.08685
