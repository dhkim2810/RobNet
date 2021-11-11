import torch

def main():
    if torch.cuda.is_available():
        print("Cuda Avail")


if __name__ == "__main__":
    main()