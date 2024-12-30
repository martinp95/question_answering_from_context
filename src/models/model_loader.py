from typing import Optional
from transformers import BertForQuestionAnswering


class ModelLoader:
    """
    A class to load and provide a pre-trained BERT model for question answering.

    This class is responsible for loading a specified pre-trained BERT model
    for question answering and providing access to it. If no model name is provided,
    it defaults to 'bert-large-uncased-whole-word-masking-finetuned-squad'.
    """

    def __init__(self, model_name: Optional[str] = 'bert-large-uncased-whole-word-masking-finetuned-squad') -> None:
        """
        Initializes the ModelLoader with a default model if no model name is provided.

        Args:
            model_name (Optional[str]): The name of the pre-trained model to load.
                Defaults to 'bert-large-uncased-whole-word-masking-finetuned-squad'.
        """
        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the BERT model for question answering.
        """
        self.model = BertForQuestionAnswering.from_pretrained(self.model_name)

    def get_model(self) -> Optional[BertForQuestionAnswering]:
        """
        Returns the loaded model.

        Returns:
            Optional[BertForQuestionAnswering]: The loaded BERT model or None if no model is loaded.
        """
        return self.model
