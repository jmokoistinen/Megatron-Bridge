from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def compare_weights(model1, model2):
    for name, param in model1.named_parameters():
        if name in model2.state_dict():
            if not torch.allclose(param, model2.state_dict()[name]):
                print(f"{name} is different")
            else:
                print(f"{name} is the same")
        else:
            print(f"{name} is not in model2")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    args = parser.parse_args()
    model1 = AutoModelForCausalLM.from_pretrained(args.model1)
    model2 = AutoModelForCausalLM.from_pretrained(args.model2)
    compare_weights(model1, model2)