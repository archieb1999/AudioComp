
# Audio Compression and Decompression with Encodec

This repository contains a Python script named `acomp.py` for encoding and decoding audio using the Encodec neural audio codec developed by Facebook Research. This implementation leverages the open-source 24kHz (for mono) and 48kHz (for stereo) models provided by Facebook Research for handling both monophonic and stereophonic audio.

## Introduction

Encodec is a state-of-the-art audio processing tool designed for efficient and high-quality audio compression and decompression. The script provided in this repository uses pre-trained Encodec models that can process both monophonic (single channel) and stereophonic (dual channel) audio across various bandwidths to either compress or decompress audio files effectively.

## Requirements

The script requires Python 3.8 and a recent version of PyTorch (ideally 1.11.0 or newer). Dependencies include:

- `torch`
- `torchaudio`

### Installation

You need to install Python on your system if it is not already installed. To install the required Python packages and the Encodec codec, use the following commands:

```bash
# Install PyTorch and torchaudio
pip install torch torchaudio

# Install Encodec from the stable release
pip install -U encodec

# Or, for the bleeding edge version directly from the repository
pip install -U git+https://git@github.com/facebookresearch/encodec#egg=encodec

# If you have cloned the Encodec repository locally
cd path/to/encodec
pip install .
```

## Usage

`acomp.py` can be run in two modes: encoding and decoding.

### Encoding

Converts an input audio file into a compressed embedding text file, with an optional output of an audio file for verification.

**Command Format**:
```bash
python acomp.py encode --input_file [path_to_input_file] --embedding_file [path_to_output_embedding_file] --bandwidth [bandwidth] [--output_file [path_to_output_audio_file]]
```

**Example**:
```bash
python acomp.py encode --input_file "example.wav" --embedding_file "example.txt" --bandwidth 6
```

### Decoding

Converts the embedding text file back into an audio file.

**Command Format**:
```bash
python acomp.py decode --embedding_file [path_to_embedding_file] --output_file [path_to_output_audio_file]
```

**Example**:
```bash
python acomp.py decode --embedding_file "example.txt" --output_file "reconstructed_example.wav"
```

### Permitted Bandwidths

- **Monophonic Audio**: 1.5, 3, 6, 12, 24 kbps
- **Stereophonic Audio**: 3, 6, 12, 24 kbps

### Resampling

Audio is resampled to match the model's expected sample rate (24kHz for monophonic and 48kHz for stereophonic). This ensures the model processes the audio correctly.

### Embedding File Format

The embedding file stores the encoded audio in a structured text format:
- The first line specifies the number of audio channels (1 for mono, 2 for stereo).
- The second line specifies the used bandwidth.
- Subsequent lines contain the embedding indices for each time step.

The structured format ensures that the embeddings can be efficiently processed for decoding, maintaining fidelity to the original audio characteristics.

## Acknowledgements

This implementation uses the Encodec neural audio codec and models developed and open-sourced by Facebook Research. The availability of these models facilitates experimentation with cutting-edge audio compression technologies in a research and development setting.

