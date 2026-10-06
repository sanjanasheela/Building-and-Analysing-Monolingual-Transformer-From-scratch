import numpy as np


class SingleBinNumPyDataset:
    """Maps input and target sequences directly from a single .bin file using pure

    NumPy.
    """

    def __init__(self, bin_path, context_length, batch_size=2):
        self.context_length = context_length
        self.batch_size = batch_size

        # Memory-map the single binary file
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.length = len(self.data) - context_length

        if self.length <= 0:
            raise ValueError(
                f"Binary file is too short for context_length {context_length}."
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Extract chunk of length context_length + 1
        chunk = self.data[idx : idx + self.context_length + 1]

        # Map to input and target arrays
        x = chunk[:-1].astype(np.int64)
        y = chunk[1:].astype(np.int64)

        return x, y

    def get_batch(self, idx):
        """Fetches a batch of sequences starting at index idx."""
        end_idx = min(idx + self.batch_size, self.length)
        current_batch_size = end_idx - idx

        inputs = np.empty(
            (current_batch_size, self.context_length), dtype=np.int64
        )
        targets = np.empty(
            (current_batch_size, self.context_length), dtype=np.int64
        )

        for i in range(current_batch_size):
            x, y = self.__getitem__(idx + i)
            inputs[i] = x
            targets[i] = y

        return inputs, targets


if __name__ == "__main__":
    BIN_PATH = "token_shards/train_shard_000.bin"
    CONTEXT_LENGTH = 16
    BATCH_SIZE = 2

    dataset = SingleBinNumPyDataset(
        bin_path=BIN_PATH, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE
    )

    # Get the first batch manually
    inputs, targets = dataset.get_batch(0)

    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)
    print("-" * 50)

    for i in range(inputs.shape[0]):
        print(f"Sample {i + 1}:")
        print(f"Input:  {inputs[i].tolist()}")
        print(f"Target: {targets[i].tolist()}")
        print()