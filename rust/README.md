# Peanut Lip Sync

Peanut Lip Sync allows you to create 2D mouth animations from voice recordings
or artificially generated voices. It analyzes your audio files, decides which
mouth shapes are appropriate, then generates lip sync information. You can use
it to animate characters for computer games, create animated content, or create
talking avatars.

Currently Peanut Lip Sync is a standalone tool but integrations are planned for
other software applications.

## Demo Video

## Mouth Shapes

Peanut Lip Sync uses 12 mouth shapes. The first 6 are the *basic mouth shapes*.
The remaining 6 are the *extended mouth shapes*.

| Label | Description | Alternative |
| ----- | ----------- | ----------- |
| A     | Closed mouth for "B", "P", "M" | |
| B     | Slightly open mouth with teeth closed. For consonants like "D", "K", "T". | |
| C     | Open mouth. Used for vowels like "EH". | | 
| D     | Wide open mouth. Used for vowels like "AH". | |
| E     | Rounded open mouth. Used for vowels like "OH". | |
| F     | Puckered lips. Use for "W" and "OO" sounds. | |
| G     | Bottom lip touching upper teeth. For "F" and "V" sounds. | B |
| H     | Tongue touching upper teeth. For "L" sound. | C |
| X     | Idle position. Resting mouth for silence. | A |
| I     | Wide mouth showing teeth for "EE" sound. | B |
| J     | Shape for "S", "CH", "J", "SH", "Z". | B |
| K     | Rounded mouth with teeth closed for "R" sound. | E |

If you don't have all the extended shapes available then you can copy
from the basic shapes. The closest basic shape is given in the table
above under "Alternative".

Peanut Lip Sync will always output all the mouth shapes.

## Command Line Options

