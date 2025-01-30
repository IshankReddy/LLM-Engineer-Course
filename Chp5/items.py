from typing import Optional
from transformers import AutoTokenizer
import re

# Define the base model to use for tokenization
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

# Token count constraints
MIN_TOKENS = 150  # Minimum required tokens to consider the content useful
MAX_TOKENS = 160  # Maximum allowed tokens before truncation

# Character length constraints
MIN_CHARS = 300  # Minimum character count to be considered valid
CEILING_CHARS = MAX_TOKENS * 7  # Maximum characters allowed before truncation

class Item:
    """
    Represents a cleaned, curated datapoint of a product with a price.
    """
    
    # Load the tokenizer for processing text
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Constants for prompt creation
    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"
    
    # List of phrases and words to be removed from details
    REMOVALS = [
        '"Batteries Included?": "No"', '"Batteries Included?": "Yes"',
        '"Batteries Required?": "No"', '"Batteries Required?": "Yes"',
        "By Manufacturer", "Item", "Date First", "Package", ":",
        "Number of", "Best Sellers", "Number", "Product "
    ]
    
    # Instance attributes
    title: str  # Product title
    price: float  # Product price
    category: str  # Product category
    token_count: int = 0  # Count of tokens in processed text
    details: Optional[str]  # Additional product details
    prompt: Optional[str] = None  # Generated prompt for training
    include = False  # Flag to determine if this item should be included

    def __init__(self, data, price):
        """
        Initialize an Item instance with product data and price.
        """
        self.title = data['title']
        self.price = price
        self.parse(data)

    def scrub_details(self):
        """
        Cleans up the details string by removing predefined unwanted phrases.
        """
        details = self.details
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details

    def scrub(self, stuff):
        """
        Cleans up the input text by removing unnecessary characters and whitespace.
        Also removes words that are 7+ characters long and contain numbers,
        as these are likely irrelevant product numbers.
        """
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,", ",").replace(",,", ",")
        words = stuff.split(' ')
        select = [word for word in words if len(word) < 7 or not any(char.isdigit() for char in word)]
        return " ".join(select)
    
    def parse(self, data):
        """
        Parses the input data and determines if it meets the criteria for inclusion.
        If valid, generates a tokenized version and sets the include flag.
        """
        # Combine different product details into a single content string
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'
        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'
        
        # Check if the content meets the minimum character requirement
        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]  # Truncate if needed
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            
            # Tokenize and check token length
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]  # Truncate if necessary
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True  # Mark as valid for inclusion

    def make_prompt(self, text):
        """
        Creates a formatted prompt for training purposes.
        """
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        Returns a test prompt with the actual price removed for evaluation.
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self):
        """
        Returns a string representation of the Item object.
        """
        return f"<{self.title} = ${self.price}>"
