import torch
import numpy as np

def estimate_total_gpu_usage(model, data_dict, in_bytes=False):
    model_size = 0
    for param in model.parameters():
        model_size += np.prod(param.size()) * param.element_size()
    for buffer in model.buffers():
        model_size += np.prod(buffer.size()) * buffer.element_size()
    
    input_keys = ['enc_tokens', 'dec_tokens', 'enc_mask', 'dec_mask']
    input_size = sum(np.prod(data_dict[key].size()) * data_dict[key].element_size() for key in input_keys if key in data_dict)
    
    total_size = model_size + input_size
    
    if in_bytes:
        return total_size
    else:
        return total_size / (1024 ** 2)

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def greedy_decode(model, batch: dict, config: dict, dec_tokenizer):
    # Encode the input
    enc_out = model.encode(batch['enc_tokens'], batch['enc_mask'])

    bos_token_id = dec_tokenizer.token_to_id('[SOS]')
    eos_token_id = dec_tokenizer.token_to_id('[PAD]')

    dec_input = torch.tensor([bos_token_id]).unsqueeze(0)

    while dec_input.size(1) != config['max_seq_len']:

        dec_mask = causal_mask(dec_input.size(1)).int()
        dec_out = model.decode(dec_input, enc_out, enc_out, batch['enc_mask'], dec_mask)
        prob = torch.softmax(model.project(dec_out[:, -1]), dim=1)
        _, next_word = torch.max(prob, dim=1)
        dec_input = torch.cat((
            dec_input,
            next_word.unsqueeze(0)
        ), dim=1)

        if next_word == eos_token_id:
            break
    
    return dec_input.squeeze(0)