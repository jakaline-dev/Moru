import random
from torch.utils.data import DataLoader, BatchSampler
from .custom_dataset import CustomDataset
from .utils import convert_to_buckets


class BucketBatchSampler(BatchSampler):
    def __init__(self, buckets, batch_size: int = 1, drop_last: bool = False):
        self.buckets = buckets
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batches = []
        bucket_keys = list(self.buckets.keys())
        random.shuffle(bucket_keys)

        for key in bucket_keys:
            batches = []
            data_list = self.buckets[key]
            random.shuffle(data_list)
            for d in data_list:
                batches.append(d)
                if len(batches) == self.batch_size:
                    yield batches
                    batches = []
            if not self.drop_last and len(batches) > 0:
                yield batches

    def __len__(self):
        total_batches = 0
        for key in self.buckets.keys():
            total_batches += len(self.buckets[key]) // self.batch_size
            if len(self.buckets[key]) % self.batch_size > 0 and not self.drop_last:
                total_batches += 1
        return total_batches


def get_bucket_dataloader(
    dataset: CustomDataset,
    batch_size: int = 1,
    drop_last: bool = False,
    pin_memory: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
):
    buckets = convert_to_buckets(dataset.data)
    # print({key: len(value) for key, value in buckets.items()})
    batch_sampler = BucketBatchSampler(
        buckets, batch_size=batch_size, drop_last=drop_last
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
