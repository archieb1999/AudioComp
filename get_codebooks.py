import torch
from encodec import EncodecModel

def get_codebooks(audio_type):
    """
    Load the Encodec model based on the audio type and concatenate the codebooks.

    Parameters:
        audio_type (str): Type of audio model to load. 
                          Choose 'mono'/'encodec_24khz' for 24kHz model, 
                          or 'stereo'/'encodec_48khz' for 48kHz model.

    Returns:
        torch.Tensor: A 3D tensor containing concatenated codebooks with the shape 
                      [codebook_level, num_vectors_per_codebook, vector_dimension].
    """
    # Map the audio type to the corresponding model
    if audio_type in ('mono', 'encodec_24khz'):
        model = EncodecModel.encodec_model_24khz(pretrained=True)
    elif audio_type in ('stereo', 'encodec_48khz'):
        model = EncodecModel.encodec_model_48khz(pretrained=True)
    else:
        raise ValueError("Unsupported audio type. Choose 'mono', 'encodec_24khz', 'stereo', or 'encodec_48khz'.")

    # Access the quantizer
    quantizer = model.quantizer

    # Collect all codebook tensors from each level of the quantizer
    codebook_tensors = [layer._codebook.embed for layer in quantizer.vq.layers]

    # Stack all the tensors along a new dimension to create a 3D tensor
    # Resulting shape will be [codebook_level, num_vectors_per_codebook, vector_dimension]
    concatenated_codebooks = torch.stack(codebook_tensors, dim=0)

    return concatenated_codebooks
