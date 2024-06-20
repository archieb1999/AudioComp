import argparse
import torch
import torchaudio
from encodec.model import EncodecModel
from typing import List, Tuple, Optional

# Define the type for EncodedFrame
EncodedFrame = Tuple[torch.Tensor, Optional[torch.Tensor]]

def load_audio(file_path):
    """Load audio file."""
    audio, sr = torchaudio.load(file_path)
    return audio, sr

def get_model_and_sr(channels, bandwidth):
    """Select the model based on the number of audio channels and bandwidth."""
    if channels == 1:
        if bandwidth not in [1.5, 3, 6, 12, 24]:  # Define acceptable bandwidths for mono
            raise ValueError("Unsupported bandwidth for mono audio.")
        model = EncodecModel.encodec_model_24khz(pretrained=True)
        target_sr = 24000
    elif channels == 2:
        if bandwidth not in [3, 6, 12, 24]:  # Define acceptable bandwidths for stereo
            raise ValueError("Unsupported bandwidth for stereo audio.")
        model = EncodecModel.encodec_model_48khz(pretrained=True)
        target_sr = 48000
    else:
        raise ValueError("Unsupported number of channels: Encodec supports only mono or stereo.")
    return model, target_sr

def resample_audio(audio, original_sr, target_sr):
    """Resample audio to the target sample rate if necessary."""
    if original_sr != target_sr:
        audio = torchaudio.transforms.Resample(original_sr, target_sr)(audio)
    return audio

def process_audio(input_file, bandwidth, embedding_file, output_file=None):
    """
    Process an audio file to encode it to embeddings and optionally decode it back.

    Args:
        input_file (str): Path to the input audio file.
        bandwidth (float): Bandwidth to be used for the model.
        embedding_file (str): Path where embeddings should be saved.
        output_file (str, optional): Path to save the decoded audio. If None, no audio is saved.

    """
    # Load the input audio file
    audio, sr = load_audio(input_file)
    
    # Determine the number of channels
    channels = audio.shape[0]
    
    # Get the appropriate model and target sample rate
    model, target_sr = get_model_and_sr(channels, bandwidth)
    
    # Resample the audio if necessary
    audio = resample_audio(audio, sr, target_sr)
    
    # Set the target bandwidth for the model
    model.set_target_bandwidth(bandwidth)
    
    # Convert the audio to embeddings
    with torch.no_grad():
        # Reshape the audio tensor to [batch, channels, time steps]
        audio = audio.unsqueeze(0)  # Add batch dimension
        
        # Encode the audio and obtain the embedding indices
        encoded_frames = model.encode(audio)
        
        # Extract the embedding indices from each encoded frame
        embedding_indices = []
        for frame, _ in encoded_frames:
            embedding_indices.append(frame.cpu().numpy())
        
        # Save the number of channels and bandwidth to the embedding file
        with open(embedding_file, 'w') as f:
            f.write(f"{channels}\n")
            f.write(f"{bandwidth}\n")
            for indices in embedding_indices:
                for idx in range(indices.shape[1]):
                    f.write(' '.join(map(str, indices[0, idx])) + '\n')
        
        # Decode the embeddings back to audio if an output file is provided
        if output_file:
            output_audio = model.decode(encoded_frames)
            # Save the output audio file
            torchaudio.save(output_file, output_audio[0], target_sr)
            print(f"Output audio saved to: {output_file}")
    
    print(f"Embeddings saved to: {embedding_file}")

def read_embeddings_from_txt(file_path: str, num_codebooks: int) -> List[EncodedFrame]:
    """
    Reads embeddings from a .txt file and converts them to a decodable format.

    Args:
        file_path (str): Path to the .txt file containing the embeddings.
        num_codebooks (int): Number of codebooks used in the model.

    Returns:
        List[EncodedFrame]: List of encoded frames suitable for decoding.
    """
    encoded_frames: List[EncodedFrame] = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize a list to store the codebook tensors for each frame
    codebook_list = []
    
    # Skip the first two lines (channels and bandwidth)
    for line in lines[2:]:
        if line.strip() == '':
            continue  # Skip empty lines
        
        # Split the line into integers
        codebook_values = list(map(int, line.strip().split()))
        
        # Convert the list of integers into a tensor and append to the list
        codebook_tensor = torch.tensor(codebook_values, dtype=torch.long)
        codebook_list.append(codebook_tensor)
        
        # Once we have collected all codebooks for a frame, construct the frame tensor
        if len(codebook_list) == num_codebooks:
            # Stack the codebooks to create the frame tensor of shape [K, T_f]
            frame_tensor = torch.stack(codebook_list, dim=0).unsqueeze(0)  # Add batch dimension
            encoded_frames.append((frame_tensor, None))  # Append the frame with no scale
            codebook_list = []  # Reset the list for the next frame

    return encoded_frames

def decode_audio_from_embeddings(embedding_file, output_file):
     """
    Process an audio file to encode it to embeddings and optionally decode it back.

    Args:
        input_file (str): Path to the input audio file.
        bandwidth (float): Bandwidth to be used for the model.
        embedding_file (str): Path where embeddings should be saved.
        output_file (str, optional): Path to save the decoded audio. If None, no audio is saved.

    """
    # Read the number of channels and bandwidth from the text file
    with open(embedding_file, 'r') as f:
        lines = f.readlines()
        channels = int(lines[0].strip())
        bandwidth = float(lines[1].strip())  # Read bandwidth as float
    
    # Get the appropriate model and target sample rate
    model, target_sr = get_model_and_sr(channels, bandwidth)
    
    # Set the target bandwidth for the model
    model.set_target_bandwidth(bandwidth)
    
    # Determine the number of codebooks from the model
    num_codebooks = int(bandwidth*4/(3*channels)) #1.5 kbps- 2 codebooks, 3 kbps- 4 codebooks,... For stereo it's half of mono 
    
    # Read embeddings from the txt file
    encoded_frames = read_embeddings_from_txt(embedding_file, num_codebooks)
    
    # Decode the audio from the encoded frames
    with torch.no_grad():
        output_audio = model.decode(encoded_frames)
    
    # Save the output audio file
    torchaudio.save(output_file, output_audio[0], target_sr)
    
    print(f"Decoded audio saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Audio Encoding and Decoding with Encodec')
    parser.add_argument('mode', choices=['encode', 'decode'], help='Choose between encoding and decoding mode')
    parser.add_argument('--input_file', type=str, help='Path to the input audio file')
    parser.add_argument('--embedding_file', type=str, help='Path to the embedding text file')
    parser.add_argument('--output_file', type=str, nargs='?', default=None, help='Optional path to the output audio file')
    parser.add_argument('--bandwidth', type=float, default=6, help='Bandwidth for encoding (default: 6)')
    
    args = parser.parse_args()
    
    if args.mode == 'encode':
        if not args.input_file or not args.embedding_file:
            parser.error('Encoding mode requires --input_file and --embedding_file arguments')
        process_audio(args.input_file, args.bandwidth, args.embedding_file, args.output_file)
    elif args.mode == 'decode':
        if not args.embedding_file or not args.output_file:
            parser.error('Decoding mode requires --embedding_file and --output_file arguments')
        decode_audio_from_embeddings(args.embedding_file, args.output_file)

if __name__ == '__main__':
    main()

