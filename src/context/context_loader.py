from typing import Optional

class ContextLoader:
    """
    A class to load and provide context for the question-answering system.
    
    This class is responsible for loading context from a file or setting it manually,
    and providing access to the current context.
    """
    
    def __init__(self) -> None:
        """
        Initializes the ContextLoader with an empty context.
        """
        self.context: Optional[str] = None

    def load_context_from_file(self, file_path: str) -> None:
        """
        Loads context from a specified file.

        Args:
            file_path (str): The path to the file containing the context.
        """
        with open(file_path, 'r') as file:
            self.context = file.read()

    def set_context(self, context: str) -> None:
        """
        Sets the context manually.

        Args:
            context (str): The context to be set.
        """
        self.context = context

    def get_context(self) -> Optional[str]:
        """
        Returns the current context.

        Returns:
            Optional[str]: The current context or None if no context is set.
        """
        return self.context