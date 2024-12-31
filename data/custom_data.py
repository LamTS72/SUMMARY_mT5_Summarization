from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
from configs.summary_config import ConfigDataset, ConfigModel
class CustomDataset():
    def __init__(self,
                 path_dataset=ConfigDataset.PATH_DATASET,
                 revision=ConfigDataset.REVISION,
                 lang1=ConfigDataset.LANG1,
                 lang2=ConfigDataset.LANG2,
                 train_size=ConfigModel.TRAIN_SIZE,
                 flag_info=True
                ):
      self.raw_data = self.split_dataset(train_size,
                                            path_dataset,
                                            lang1,
                                            lang2,
                                            revision)
      self.size = len(self.raw_data["train"]) + len(self.raw_data["test"]) + len(self.raw_data["validation"])
      if flag_info:
        print("-"*50, "Information of Dataset", "-"*50)
        print(self.raw_data)
        print("-"*50, "Information of Dataset", "-"*50)

    def split_dataset(self, train_size, path_dataset, lang1, lang2, revision):
      raw_data_en = load_dataset(path_dataset + "/en")
      raw_data_es = load_dataset(path_dataset + "/es")
      raw_data_en = raw_data_en.filter(self.filter_books)
      raw_data_es = raw_data_es.filter(self.filter_books)
      raw_dataset = DatasetDict()
      for key in raw_data_en.keys():
        raw_dataset[key] = concatenate_datasets(
            [raw_data_en[key], raw_data_es[key]],
        )
        raw_dataset[key] = raw_dataset[key].shuffle(seed=42)
      raw_dataset.filter(lambda x: len(x["review_title"].split()) > 2)
      # raw_data = DatasetDict(
      #     {
      #         "train": raw_dataset["train"].shuffle(seed=42).select(range(1000)),
      #         "validation": raw_dataset["validation"].shuffle(seed=42).select(range(100)),
      #         "test": raw_dataset["test"].select(range(100)),
      #     }
      # )
      return raw_dataset #raw_dataset

    def filter_books(self, example):
      return (
          example["product_category"] == "book" or
          example["product_category"] == "digital_ebook_purchase"
      )

    def __len__(self):
      return self.size

    def __getitem__(self, index):
      dataset = self.raw_data["train"]
      data = dataset[index]["review_body"]
      target = dataset[index]["review_title"]
      return {
          "data_text": data,
          "data_label": target
      }

