# Cherry Lip Sync Model

The Cherry Lip Sync model was developed based on the ideas from _Real-Time Lip
Sync for Live 2D Animation_[^1].

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
desired framerate (30fps by default). The command line interface has an optional
filtering step that can prevent single frame visemes (disabled by default).

Some of the notable things NOT present:
1) There is no Mel spectrum normalization beyond batch norm
2) There is no MFCC computation
3) The model is unidirectional, does not need to know all future audio
4) The lookahead is limited to 3 frames (compared to 6 in reference[^1])
5) The model does not need to know text or phonemes in audio

## Visualization

![model_simple onnx](https://github.com/user-attachments/assets/a2b4a1fe-fef4-45c6-9a6a-7864a85a07de)


## Training Data

The training set is a private lip sync dataset generated using audio from
LibriSpeech[^2]. The lip sync timing information was generated using a variety
of methods:

1) Manual annotation
2) Existing lip sync tools with manual quality review
3) Time warping audio and sync data in parallel for existing training examples
   to generate new examples at different cadences
4) Mapping and time warping existing visemes from a word to other audio
   instances of the same word
4) Synthetic text-to-speech generation with known phoneme timings and rules for
   visemes
5) Apply voice changing to existing audio examples without changing timing

The synthetic examples for (4) were generated using
[MeloTTS](https://github.com/myshell-ai/MeloTTS) with modifications to output
phoneme timing information. The phonemes are mapped to plausible visemes using
an ad-hoc (and evolving) set of programmatic rules with some element of random
choices. These were generated in bulk then manually reviewed quickly (e.g. 1
hour of review to generate 5 minutes of usable training data). The rules are
inspired from code in [Rhubarb](https://github.com/DanielSWolf/rhubarb-lip-sync)
and [LazyKH](https://github.com/carykh/lazykh/).

Synthetic voice changing is used to change voices without altering timing of
phonemes to extend the training set. This lets each high-quality lip sync
example in the training set be extended to multiple speaking voices. I used
[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for
voice changing.

The current training dataset contains about 1 hour of English language audio and
takes about 5 minutes to fully train with 200 epochs on an NVIDIA GTX 3090.

I'm working on cleaning up the dataset before releasing it publicly. The current
aggregated dataset is basically an experimental concatenation of a bunch of
half-baked evolving ideas. Whenever I saw something reasonable I threw it in the
training set and went back to tweaking stuff and trying more samples.

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


[^1]: D. Aneja, W. Li. _Real-Time Lip Sync for Live 2D Animation_.
    https://arxiv.org/abs/1910.08685

[^2]: V. Panayotov, G. Chen, D. Povey and S. Khudanpur, _Librispeech: An ASR
    corpus based on public domain audio books_, 2015 IEEE International
    Conference on Acoustics, Speech and Signal Processing (ICASSP), South
    Brisbane, QLD, Australia, 2015, pp. 5206-5210, doi:
    10.1109/ICASSP.2015.7178964.
