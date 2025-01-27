# Cherry Lip Sync

![Logo of lips](./images/Logo.png)

Cherry Lip Sync allows you to create 2D mouth animations from voice recordings
or artificially generated voices. It analyzes your audio files, decides which
mouth shapes are appropriate, then generates lip sync information. You can use
it to animate characters for computer games, create animated content, or create
talking avatars.

Currently Cherry Lip Sync is a standalone tool but integrations are planned for
other software applications.

## Demo Video

<video controls width="500" src="./demo/Demo.mp4" alt="Demonstration video showing talking lips to public domain audio clips"></video>

The above video demonstrates lip sync output driving mouth pictures using public
domain audio clips with no text provided. Animation sequences were directly from
Cherry Lip Sync without additional editing.

## Mouth Shapes

Cherry Lip Sync uses 12 mouth shapes. The first 6 are the *basic mouth shapes*.
The remaining 6 are the *extended mouth shapes*.

| Label | Description | Alternative |
| ----- | ----------- | ----------- |
| A     | Closed mouth for "B", "M", "P" | |
| B     | Slightly open mouth with teeth closed. For consonants like "D", "K", "T". | |
| C     | Open mouth. Used for vowels like "EH". | | 
| D     | Wide open mouth. Used for vowels like "AH". | |
| E     | Rounded open mouth. Used for vowels like "OH". | |
| F     | Puckered lips. Use for "W" and "OO" sounds. | |
| G     | Bottom lip touching upper teeth. For "F" and "V" sounds. | B |
| H     | Tongue touching upper teeth. For "L" sound. | C |
| I     | Wide mouth showing teeth for "EE" sound. | B |
| J     | Shape for "S", "CH", "J", "SH", "Z". | B |
| K     | Rounded mouth with teeth closed for "R" sound. | E |
| X     | Idle position. Resting mouth for silence. | A |

Cherry Lip Sync will always output all the mouth shapes. If you don't have all
the extended shapes available then you can copy from the basic shapes. The
closest basic shape is given in the table above under "Alternative".

## Command Line Options



## Input Format

Input audio can be in `.mp3`, `.ogg`, `.wav`, and `.flac` formats. Internally
all audio is converted to 32-bit mono at 16 kHz for analysis. If you are having
trouble getting things working or experience audio timing issues you can
manually convert your audio to 16 kHz mono in `.wav` format to rule out any
problems with the input conversion.

The lip sync model used by Cherry Lip Sync is currently solely trained on
English language audio so will probably give the best results with English
language input. Other languages that share phonemes with English will probably
give better results than languages with fewer similarities to English.

Input audio should be as clean as possible. Extraneous noises may be interpreted
as vocal sounds requiring lip movements.

## Output Format

## Comparison with Rhubarb

<video controls width="500" src="./demo/Compare.mp4" alt="Demonstration video showing Rhubarb with and without text provided versus Cherry"></video>
