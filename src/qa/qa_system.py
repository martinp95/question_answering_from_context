from typing import Optional
from transformers import BertForQuestionAnswering, BertTokenizer


class QASystem:
    """
    A class to handle question answering based on a given context using a pre-trained BERT model.

    This class uses a pre-trained BERT model and tokenizer to answer questions based on a provided context.
    It encodes the question and context, performs predictions using the model, and extracts the answer from the model's output.
    """

    def __init__(self, model: Optional[BertForQuestionAnswering], tokenizer: BertTokenizer, context: str) -> None:
        """
        Initializes the question-answering system.

        Args:
            model (Optional[BertForQuestionAnswering]): The pre-trained model for question answering.
            tokenizer (BertTokenizer): The pre-trained tokenizer corresponding to the model.
            context (str): The context in which the answers will be based.
        """
        if model is None:
            raise ValueError("Model cannot be None")
        self.model = model
        self.tokenizer = tokenizer
        self.context = context

    def ask_question(self, question: str) -> str:
        """
        Asks a question and returns the answer based on the context.

        Args:
            question (str): The question to be asked.

        Returns:
            str: The answer to the question based on the context.
        """
        # Encode the question and context into tensors
        inputs = self.tokenizer.encode_plus(
            question, self.context, add_special_tokens=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Perform the prediction using the model
        outputs = self.model(input_ids, attention_mask=attention_mask)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        # Find the start and end positions of the answer
        answer_start = answer_start_scores.argmax()
        answer_end = answer_end_scores.argmax() + 1

        # Convert the token IDs of the answer to a string
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

        return answer.strip()
