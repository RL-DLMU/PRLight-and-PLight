import torch
from EDQN import Encoder, VAnet, Decoder


def load_model_and_get_state_dict(model_path):
    model = Encoder(20, 32)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model.state_dict()


def average_state_dicts(state_dicts):
    averaged_dict = {}

    for key, value in state_dicts[0].items():
        averaged_dict[key] = torch.zeros_like(value)

    for state_dict in state_dicts:
        for key, value in state_dict.items():
            averaged_dict[key] += value

    num_models = len(state_dicts)
    for key, value in averaged_dict.items():
        averaged_dict[key] /= num_models

    return averaged_dict


def save_averaged_state_dict(averaged_state_dict, save_path):
    torch.save(averaged_state_dict, save_path)


model_paths = ["model_pool/model_1/encoder.pt", "model_pool/model_2/encoder.pt"]
state_dicts = [load_model_and_get_state_dict(path) for path in model_paths]
averaged_state_dict = average_state_dicts(state_dicts)
save_path = "model_pool/avg_encoder.pt"
save_averaged_state_dict(averaged_state_dict, save_path)