from datetime import datetime
from tqdm import tqdm  # Library for progress bars
from datasets import load_dataset  # Hugging Face dataset loader
from concurrent.futures import ProcessPoolExecutor  # For parallel processing
from items import Item  # Importing the Item class from items module

# Constants for processing
CHUNK_SIZE = 1000  # Number of items to process in one batch
MIN_PRICE = 0.5  # Minimum valid price for an item
MAX_PRICE = 999.49  # Maximum valid price for an item


class ItemLoader:
    """
    A class responsible for loading and processing dataset items.
    """

    def __init__(self, name):
        """
        Initialize an ItemLoader with a dataset name.
        
        :param name: The name of the dataset (category of items).
        """
        self.name = name  # Store dataset name
        self.dataset = None  # Placeholder for dataset, to be loaded later

    def from_datapoint(self, datapoint):
        """
        Convert a single datapoint into an Item instance.
        
        - Extracts and validates the price.
        - Creates an Item instance if price is within valid range.
        - Returns None if the price is invalid or the item does not meet criteria.

        :param datapoint: Dictionary containing product data.
        :return: Item instance if valid, else None.
        """
        try:
            price_str = datapoint['price']  # Extract price as a string
            if price_str:
                price = float(price_str)  # Convert price to float
                if MIN_PRICE <= price <= MAX_PRICE:  # Ensure price is within range
                    item = Item(datapoint, price)  # Create an Item instance
                    return item if item.include else None  # Return only if item is included
        except ValueError:  # Handle cases where price conversion fails
            return None

    def from_chunk(self, chunk):
        """
        Process a batch (chunk) of datapoints to create Item instances.
        
        :param chunk: List of datapoints from the dataset.
        :return: List of valid Item instances.
        """
        batch = []
        for datapoint in chunk:
            result = self.from_datapoint(datapoint)  # Process each datapoint
            if result:
                batch.append(result)  # Add valid items to batch
        return batch

    def chunk_generator(self):
        """
        Generator that yields chunks of data from the dataset.
        
        :yield: Chunks of data with CHUNK_SIZE elements.
        """
        size = len(self.dataset)  # Get total dataset size
        for i in range(0, size, CHUNK_SIZE):  
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))  # Yield chunks of size CHUNK_SIZE

    def load_in_parallel(self, workers):
        """
        Process dataset items in parallel using multiple workers.

        - Splits the dataset into chunks and processes them concurrently.
        - Uses ProcessPoolExecutor for multiprocessing.
        - Displays a progress bar using tqdm.

        :param workers: Number of parallel processes to use.
        :return: List of processed Item instances.
        """
        results = []  # Store processed items
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1  # Calculate total number of chunks

        with ProcessPoolExecutor(max_workers=workers) as pool:  # Create a process pool
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)  # Collect results from parallel execution

        # Assign category name to each processed item
        for result in results:
            result.category = self.name  
        return results  # Return processed items

    def load(self, workers=8):
        """
        Load and process dataset items.
        
        - Fetches data from Hugging Face datasets.
        - Processes data in parallel using multiple workers.
        - Displays total processing time.

        :param workers: Number of parallel workers for processing (default: 8).
        :return: List of processed Item instances.
        """
        start = datetime.now()  # Record start time
        print(f"Loading dataset {self.name}", flush=True)

        # Load dataset from Hugging Face
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",  # Dataset source
            f"raw_meta_{self.name}",  # Dataset category
            split="full",
            trust_remote_code=True
        )

        results = self.load_in_parallel(workers)  # Process dataset in parallel
        finish = datetime.now()  # Record end time

        # Display completion message with time taken
        print(f"Completed {self.name} with {len(results):,} datapoints in {(finish-start).total_seconds()/60:.1f} mins", flush=True)
        
        return results  # Return processed dataset items
