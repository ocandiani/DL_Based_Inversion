import os
import torch
import json
import argparse
import datetime

from collections import OrderedDict
from layers import PRIMITIVES

def load_model(path):
    """Carrega o dicionário de estado do modelo do caminho fornecido.

    Args:
        path (str): Caminho para o arquivo de checkpoint contendo o estado do modelo.

    Returns:
        OrderedDict: O dicionário de estado do modelo carregado.
    """

    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint['model']

def get_best_arch_from_state_dict(state_dict):
    """Extrai a melhor arquitetura do dicionário de estado do modelo carregado.

    Args:
        state_dict (OrderedDict): O dicionário de estado do modelo.

    Returns:
        list: Uma lista de tuplas, onde cada tupla representa um nome de bloco e
              uma lista das melhores operações para cada passo naquele bloco.
    """

    best_arch = []
    for name, param in state_dict.items():
        if 'alphas' in name:
            block_name = name.split('.alphas')[0]
            alphas = param.data
            best_ops = []
            for step_alphas in alphas:
                best_op_index = step_alphas.argmax().item()
                best_op = PRIMITIVES[best_op_index]
                best_ops.append(best_op)
            best_arch.append((block_name, best_ops))
    return best_arch

def extract_best_architecture(model_path, output_path):
    """Extrai a melhor arquitetura do dicionário de estado do modelo e salva como um arquivo JSON.

    Args:
        model_path (str): Caminho para o arquivo de checkpoint contendo o estado do modelo.
        output_path (str): Caminho para onde sera salvo o JSON de saída.
    """

    state_dict = load_model(model_path)
    best_arch = get_best_arch_from_state_dict(state_dict)

    print("Best Architecture:")
    print(json.dumps(best_arch, indent=4))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    exp_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_json_path = os.path.join(output_path,f'{exp_id}.json')
    # Save the best architecture as JSON
    with open(output_json_path, 'w') as f:
        json.dump(best_arch, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Extrai a melhor arquitetura de um modelo DARTS PyTorch e salva em um arquivo JSON.')
    parser.add_argument('-p', '--model_path', required=True, help='Caminho para o arquivo do modelo')
    parser.add_argument('-o', '--output_path', default='./darts_models', help='Caminho para o arquivo JSON de saída')
    args = parser.parse_args()

    extract_best_architecture(args.model_path, args.output_path)

if __name__ == '__main__':
    main()