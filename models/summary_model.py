from transformers import(
    AutoModelForSeq2SeqLM
)

class CustomModel():
  def __init__(self,
               model_name=None,
               flag_training=True
               ) -> None:
    self.model_name = model_name
    self.model = self.create_model()
    if flag_training:
      print("-"*50, "Information of Model", "-"*50)
      print(self.model)
      print("Parameters: ", int(self.model.num_parameters() / 1000000),  "M")
      print("-"*50, "Information of Model", "-"*50)

  def create_model(self):
    return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)