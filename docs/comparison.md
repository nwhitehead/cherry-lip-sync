# Comparison with Rhubarb

https://github.com/user-attachments/assets/80f5fa74-776c-465f-a658-2872c3748a93

The above video shows Rhubarb output for a challenging synthetic voice. On the
left is the result without providing any text. On the right is the Rhubarb
result with source text provided.

Next Rhubarb is shown against Cherry. In this comparison Rhubarb has source text
provided but Cherry just uses audio. Rhubarb struggles to get enough viseme
changes to make the animation look convincing. Cherry picks more visemes but is
perhaps too "chattery".

## Performance

On my desktop, a _HP All-in-One 24-df1xxx_ with an 8 core _11th Gen Intel® Core™
i5-1135G7_ CPU, `cheerylipsync` processes `605` seconds of audio stored in `.mp3`
format in `13.542s` (wall clock time). That is `44.7` times faster than
realtime.

`rhubarb` with _pocketSphinx_ recognizer takes `9.28s` (wall clock time) to
process an `.ogg` file of length `7.7s`. That is `0.83` times faster than
realtime (i.e. slower than realtime).

`rhubarb` with _phonetic_ recognizer takes `2.27s` to process the same `7.7s`
audio file, for a factor of `3.39` times faster than realtime. For the longer
file of `605` seconds, it takes `33.56s` for a factor of `18.03` times faster
than realtime.
