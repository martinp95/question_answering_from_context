import unittest
import sys
import os
import re
from difflib import SequenceMatcher

# Add the path to the source code directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from qa.qa_system import QASystem
from models.model_loader import ModelLoader
from tokenizer_loader import TokenizerLoader
from context import ContextLoader

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize_text(text):
    return re.sub(r'\W+', ' ', text).strip().lower()

class TestQASystem(unittest.TestCase):
    """
    Test suite for the QASystem class.
    
    This class contains unit tests for the QASystem class, which is responsible for
    answering questions based on a given context. The tests cover various aspects
    of the QASystem's functionality, including its ability to correctly answer
    questions about the location, designer, construction time, criticism, and event
    related to the Eiffel Tower.
    """
    
    def setUp(self):
        """
        Set up the test environment by loading the model, tokenizer, and context.
        
        This method initializes the context with information about the Eiffel Tower,
        loads the model and tokenizer using the ModelLoader and TokenizerLoader classes,
        and creates an instance of the QASystem class with the loaded model, tokenizer,
        and context.
        """
        self.context = (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
            "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
            "Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized "
            "by some of France's leading artists and intellectuals for its design, but it has become a global cultural "
            "icon of France and one of the most recognizable structures in the world."
        )

        context_loader = ContextLoader()
        context_loader.set_context(self.context)
        
        model_loader = ModelLoader('bert-large-uncased-whole-word-masking-finetuned-squad')
        tokenizer_loader = TokenizerLoader('bert-large-uncased-whole-word-masking-finetuned-squad')
        
        self.qa_system = QASystem(model_loader.get_model(),
                                  tokenizer_loader.get_tokenizer(),
                                  context_loader.get_context())

    def test_location_question(self):
        """
        Test the QASystem's ability to answer a question about the location of the Eiffel Tower.
        
        This test checks if the QASystem can correctly identify the location of the Eiffel Tower
        from the given context. The expected answer is "on the Champ de Mars in Paris, France".
        """
        question = "Where is the Eiffel Tower located?"
        expected_answer = "on the Champ de Mars in Paris, France"
        answer = self.qa_system.ask_question(question)
        self.assertGreater(similar(normalize_text(expected_answer), normalize_text(answer)), 0.8)

    def test_designer_question(self):
        """
        Test the QASystem's ability to answer a question about the designer of the Eiffel Tower.
        
        This test checks if the QASystem can correctly identify the designer of the Eiffel Tower
        from the given context. The expected answer is "Gustave Eiffel".
        """
        question = "Who designed the Eiffel Tower?"
        expected_answer = "Gustave Eiffel"
        answer = self.qa_system.ask_question(question)
        self.assertGreater(similar(normalize_text(expected_answer), normalize_text(answer)), 0.8)

    def test_construction_time_question(self):
        """
        Test the QASystem's ability to answer a question about the construction time of the Eiffel Tower.
        
        This test checks if the QASystem can correctly identify the construction time of the Eiffel Tower
        from the given context. The expected answer is "from 1887 to 1889".
        """
        question = "When was the Eiffel Tower constructed?"
        expected_answer = "from 1887 to 1889"
        answer = self.qa_system.ask_question(question)
        self.assertGreater(similar(normalize_text(expected_answer), normalize_text(answer)), 0.8)

    def test_criticism_reason_question(self):
        """
        Test the QASystem's ability to answer a question about the reason for the initial criticism of the Eiffel Tower.
        
        This test checks if the QASystem can correctly identify the reason for the initial criticism of the Eiffel Tower
        from the given context. The expected answer is "for its design".
        """
        question = "Why was the Eiffel Tower initially criticized?"
        expected_answer = "for its design"
        answer = self.qa_system.ask_question(question)
        self.assertGreater(similar(normalize_text(expected_answer), normalize_text(answer)), 0.8)

    def test_event_question(self):
        """
        Test the QASystem's ability to answer a question about the event for which the Eiffel Tower was constructed.
        
        This test checks if the QASystem can correctly identify the event for which the Eiffel Tower was constructed
        from the given context. The expected answer is "the 1889 World's Fair".
        """
        question = "What event was the Eiffel Tower constructed for?"
        expected_answer = "the 1889 World's Fair"
        answer = self.qa_system.ask_question(question)
        self.assertGreater(similar(normalize_text(expected_answer), normalize_text(answer)), 0.8)

if __name__ == '__main__':
    unittest.main()