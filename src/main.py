import argparse
from qa import QASystem
from models import ModelLoader
from tokenizer_loader import TokenizerLoader
from context import ContextLoader


def main():
    """
    Main function to run the Batch Question Answering System.

    Parses command-line arguments, loads the model, tokenizer, and context,
    and starts an interactive question-answering session.
    """
    parser = argparse.ArgumentParser(
        description="Batch Question Answering System")
    parser.add_argument('-m', '--model', type=str, default='bert-large-uncased-whole-word-masking-finetuned-squad',
                        help="The name of the pre-trained model to load (default: 'bert-large-uncased-whole-word-masking-finetuned-squad')")
    parser.add_argument('-t', '--tokenizer', type=str, default='bert-large-uncased-whole-word-masking-finetuned-squad',
                        help="The name of the pre-trained tokenizer to load (default: 'bert-large-uncased-whole-word-masking-finetuned-squad')")
    parser.add_argument('-cf', '--context_file', type=str,
                        help="Path to the file containing the context")
    parser.add_argument('-c', '--context', type=str,
                        help="The context string to use for answering questions")
    args = parser.parse_args()

    # Load context from file or string
    if args.context_file:
        context_loader = ContextLoader()
        context_loader.load_context_from_file(args.context_file)
    elif args.context:
        context_loader = ContextLoader()
        context_loader.set_context(args.context)
    else:
        raise ValueError("Either --context_file or --context must be provided")

    # Load model and tokenizer
    model_loader = ModelLoader(args.model)
    tokenizer_loader = TokenizerLoader(args.tokenizer)

    # Initialize the QA system
    qa_system = QASystem(model_loader.get_model(),
                         tokenizer_loader.get_tokenizer(),
                         context_loader.get_context())

    print(f"Using model: {args.model} and tokenizer: {args.tokenizer}")

    # Interactive question-answering session
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = qa_system.ask_question(question.strip())
        print(f"Question: {question.strip()}")
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
