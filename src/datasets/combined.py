from torch.utils.data import Dataset
import numpy as np

class CombinedChangeDataset(Dataset):
    def __init__(self, pscd_dataset, vl_cmu_dataset, our_dataset, s2looking_dataset):
        self.datasets = [pscd_dataset, vl_cmu_dataset, our_dataset, s2looking_dataset]
        self.lengths = [len(ds) for ds in self.datasets]
        self.cumulative_lengths = [0] + list(np.cumsum(self.lengths))

        # dataset IDs: 0 = PSCD, 1 = VL-CMU, 2 = Our, 3 = S2Looking
        self.dataset_ids = [
            [i] * length for i, length in enumerate(self.lengths)
        ]
        self.dataset_ids = np.concatenate(self.dataset_ids)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        for i in range(len(self.datasets)):
            if self.cumulative_lengths[i] <= idx < self.cumulative_lengths[i + 1]:
                dataset = self.datasets[i]
                local_idx = idx - self.cumulative_lengths[i]
                sample = dataset[local_idx]

                if len(sample) == 3:
                    t0, t1, mask = sample
                elif len(sample) == 4:
                    t0, t1, mask, caption = sample
                else:
                    raise ValueError(f"Unexpected sample length: {len(sample)}")

                return t0, t1, mask, caption

    @property
    def figsize(self):
        # All sub-datasets are resized/cropped to 512x512
        return np.array([512, 512])