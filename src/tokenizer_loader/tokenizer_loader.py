from typing import Optional
from transformers import BertTokenizer


class TokenizerLoader:
    """
    A class to load and provide a pre-trained BERT tokenizer.

    This class is responsible for loading a specified pre-trained BERT tokenizer
    and providing access to it. If no tokenizer name is provided, it defaults to
    'bert-large-uncased-whole-word-masking-finetuned-squad'.
    """

    def __init__(self, tokenizer_name: Optional[str] = 'bert-large-uncased-whole-word-masking-finetuned-squad') -> None:
        """
        Initializes the TokenizerLoader with a default tokenizer if no tokenizer name is provided.

        Args:
            tokenizer_name (Optional[str]): The name of the pre-trained tokenizer to load.
                Defaults to 'bert-large-uncased-whole-word-masking-finetuned-squad'.
        """
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self.load_tokenizer()

    def load_tokenizer(self) -> None:
        """
        Loads the specified BERT tokenizer.
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)

    def get_tokenizer(self) -> Optional[BertTokenizer]:
        """
        Returns the loaded tokenizer.

        Returns:
            Optional[BertTokenizer]: The loaded tokenizer or None if no tokenizer is loaded.
        """
        return self.tokenizer
