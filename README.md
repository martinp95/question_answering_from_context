# Batch Application for Question Answering

This project is a batch application designed to answer questions based on a given context using pre-trained models and tokenizers.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Example](#example)
- [Example Question and Answer](#example-question-and-answer)
- [Running Tests](#running-tests)

## Project Structure

```
question_answering_from_context
├── src
│   ├── __init__.py
│   ├── main.py                # Entry point of the application
│   ├── models
│   │   ├── __init__.py
│   │   └── model_loader.py    # Loads specified models
│   ├── tokenizer_loader
│   │   ├── __init__.py
│   │   └── tokenizer_loader.py # Loads specified tokenizers
│   ├── context
│   │   ├── __init__.py
│   │   └── context_loader.py   # Loads context from files or sets context directly
│   └── qa
│       ├── __init__.py
│       └── qa_system.py       # Processes questions and returns answers
├── test
│   ├── __init__.py
│   └── test_qa_system.py      # Unit tests for the QA system
├── environment.yml            # Conda environment configuration
└── README.md                  # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/martinp95/question_answering_from_context.git
   cd question_answering_from_context
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f environment.yml
   conda activate question_answering_env
   ```

## Usage

Run the application with the required parameters:
```
python src/main.py --model <model_name> --tokenizer <tokenizer_name> --context_file <path_to_context_file>
```
or
```
python src/main.py --model <model_name> --tokenizer <tokenizer_name> --context "<context_string>"
```

### Example

- Using a context file:
  ```
  python src/main.py --model bert-large-uncased-whole-word-masking-finetuned-squad --tokenizer bert-large-uncased-whole-word-masking-finetuned-squad --context_file context.txt
  ```

- Using a context string:
  ```
  python src/main.py --model bert-large-uncased-whole-word-masking-finetuned-squad --tokenizer bert-large-uncased-whole-word-masking-finetuned-squad --context "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
  ```

### Example Question and Answer

Using the default context:
```
Context: "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world."

Question: "What event was the Eiffel Tower constructed for?"
Answer: "the 1889 World's Fair"
```

## Running Tests

To run the unit tests for the QA system, use the following command:
```
python -m unittest discover -s test
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.