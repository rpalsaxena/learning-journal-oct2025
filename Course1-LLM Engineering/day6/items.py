# Import necessary libraries for type hints, tokenization, and regex operations
from typing import Optional
from transformers import AutoTokenizer
import re

# Configuration constants for the LLM fine-tuning dataset preparation

# Base model used for tokenization - Meta's Llama 3.1 8B parameter model
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

# Token count constraints for training data quality
MIN_TOKENS = 150  # Minimum tokens required - ensures we have substantial content
MAX_TOKENS = 160  # Maximum tokens before truncation - keeps prompts manageable
                  # After adding prompt text, total will be ~180 tokens

# Character count constraints for initial filtering
MIN_CHARS = 300        # Minimum characters needed before tokenization
CEILING_CHARS = MAX_TOKENS * 7  # Rough estimate: ~7 chars per token for pre-filtering

class Item:
    """
    An Item represents a cleaned, curated datapoint of a Product with a Price
    Used for preparing training data for LLM fine-tuning on price prediction tasks
    
    This class processes raw product data, cleans it, tokenizes it, and formats it
    into proper training prompts with consistent question-answer structure.
    """
    
    # Class-level tokenizer shared across all instances for efficiency
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Template strings for consistent prompt formatting
    PREFIX = "Price is $"  # Prefix for the answer part of the training prompt
    QUESTION = "How much does this cost to the nearest dollar?"  # Standard question
    
    # Common noise phrases to remove from product details
    # These don't add value for price prediction and just consume tokens
    REMOVALS = [
        '"Batteries Included?": "No"', '"Batteries Included?": "Yes"', 
        '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', 
        "By Manufacturer", "Item", "Date First", "Package", ":", 
        "Number of", "Best Sellers", "Number", "Product "
    ]

    # Instance attributes with type hints
    title: str              # Product title/name
    price: float           # Price in dollars
    category: str          # Product category
    token_count: int = 0   # Final token count of the complete prompt
    details: Optional[str] # Additional product details (may be None)
    prompt: Optional[str] = None  # Final formatted training prompt
    include: bool = False  # Whether this item passes quality filters for inclusion

    def __init__(self, data, price):
        """
        Initialize an Item from raw product data and price
        
        Args:
            data: Dictionary containing product information (title, description, features, details)
            price: Float representing the product price in dollars
        """
        self.title = data['title']
        self.price = price
        self.parse(data)  # Process and validate the data

    def scrub_details(self):
        """
        Clean up the details string by removing common text that doesn't add value
        
        Removes boilerplate phrases and common product metadata that don't help
        with price prediction but consume valuable tokens in the training data.
        
        Returns:
            str: Cleaned details string with noise phrases removed
        """
        details = self.details
        # Remove each noise phrase from the REMOVALS list
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details

    def scrub(self, stuff):
        """
        Clean up the provided text by removing unnecessary characters and whitespace
        Also remove words that are 7+ chars and contain numbers (likely product codes/SKUs)
        
        This aggressive cleaning helps focus the model on descriptive content rather
        than technical specifications that don't correlate well with price.
        
        Args:
            stuff: Raw text string to clean
            
        Returns:
            str: Cleaned text with normalized whitespace and filtered words
        """
        # Remove punctuation, brackets, and normalize whitespace
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        
        # Clean up comma formatting
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        
        # Split into words and filter out long words containing numbers
        # These are often model numbers, SKUs, or technical specs that don't help price prediction
        words = stuff.split(' ')
        select = [word for word in words if len(word)<7 or not any(char.isdigit() for char in word)]
        return " ".join(select)
    
    def parse(self, data):
        """
        Parse and validate this datapoint for inclusion in training dataset
        
        Combines description, features, and details into a single text block,
        applies cleaning, tokenizes, and checks if it meets quality thresholds.
        Only items that pass all filters get included in the final dataset.
        
        Args:
            data: Dictionary containing 'description', 'features', and 'details' keys
        """
        # Combine description lines into a single string
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'
            
        # Add features section if present
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'
            
        # Add cleaned details section if present
        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'
        
        # Apply quality filters
        if len(contents) > MIN_CHARS:  # Must have minimum character count
            # Truncate if too long (rough pre-filtering before tokenization)
            contents = contents[:CEILING_CHARS]
            
            # Apply text cleaning to both title and content
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            
            # Tokenize and check token count
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) > MIN_TOKENS:  # Must have sufficient token content
                # Truncate to maximum allowed tokens
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                
                # Create the final training prompt and mark for inclusion
                self.make_prompt(text)
                self.include = True

    def make_prompt(self, text):
        """
        Create a formatted training prompt in question-answer format
        
        Formats the cleaned product text into a consistent training example:
        - Question asking for price
        - Product information 
        - Answer with exact price
        
        This creates supervised learning data where the model learns to predict
        prices based on product descriptions.
        
        Args:
            text: Cleaned and tokenized product description text
        """
        # Build the prompt: Question + Product Info + Answer
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"  # Round to nearest dollar
        
        # Calculate final token count for the complete prompt
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        Return a prompt suitable for testing/inference, with the actual price removed
        
        This creates the input portion that would be fed to a trained model
        for price prediction, excluding the answer portion used during training.
        
        Returns:
            str: Question and product info without the price answer
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self):
        """
        Return a readable string representation of this Item
        
        Returns:
            str: Formatted string showing title and price for debugging
        """
        return f"<{self.title} = ${self.price}>"

