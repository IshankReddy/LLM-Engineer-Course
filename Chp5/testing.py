import math
import matplotlib.pyplot as plt  # Library for data visualization

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# Mapping colors to their ANSI codes
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}


class Tester:
    """
    A class to evaluate a predictor function against a dataset.
    It calculates errors, logs predictions, and generates a visual report.
    """

    def __init__(self, predictor, data, title=None, size=250):
        """
        Initialize the Tester class.

        :param predictor: The function used to predict prices.
        :param data: The dataset containing ground truth values.
        :param title: Optional title for the report.
        :param size: Number of data points to test (default: 250).
        """
        self.predictor = predictor  # Function to make predictions
        self.data = data  # Dataset containing actual price values
        self.title = title or predictor.__name__.replace("_", " ").title()  # Set default title
        self.size = size  # Number of datapoints to test
        self.guesses = []  # Stores predicted values
        self.truths = []  # Stores actual values
        self.errors = []  # Stores absolute error values
        self.sles = []  # Stores squared log errors
        self.colors = []  # Stores color-coded error categories

    def color_for(self, error, truth):
        """
        Determine the color category based on the error magnitude.

        - Green: Low error (<40 or <20% of true value).
        - Orange: Medium error (<80 or <40% of true value).
        - Red: High error.

        :param error: The absolute prediction error.
        :param truth: The actual price.
        :return: Color category as a string.
        """
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i):
        """
        Run a single data point through the predictor and log results.

        - Computes the absolute error and squared log error (SLE).
        - Categorizes the error using color coding.
        - Prints the result with color-coded formatting.

        :param i: Index of the data point in the dataset.
        """
        datapoint = self.data[i]  # Get data point
        guess = self.predictor(datapoint)  # Predict price
        truth = datapoint.price  # Get actual price
        error = abs(guess - truth)  # Compute absolute error
        log_error = math.log(truth + 1) - math.log(guess + 1)  # Compute log error
        sle = log_error ** 2  # Compute squared log error
        color = self.color_for(error, truth)  # Determine error category

        # Format title for readability (truncate long titles)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40] + "..."

        # Store results
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)

        # Print color-coded results
        print(
            f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} "
            f"Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}"
        )

    def chart(self, title):
        """
        Generate a scatter plot comparing true prices vs. predicted prices.

        - Uses matplotlib to create the visualization.
        - Colors points based on the error category.

        :param title: Title for the plot.
        """
        max_error = max(self.errors)  # Get maximum error
        plt.figure(figsize=(12, 8))  # Set figure size
        max_val = max(max(self.truths), max(self.guesses))  # Get max value for scaling

        # Plot reference line (ideal predictions)
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)

        # Scatter plot of actual vs. predicted values
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)

        # Set labels and title
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)

        # Display the plot
        plt.show()

    def report(self):
        """
        Compute overall error metrics and generate the final report.

        - Calculates the average absolute error.
        - Computes Root Mean Squared Logarithmic Error (RMSLE).
        - Determines the percentage of "green" (accurate) predictions.
        - Generates a scatter plot of results.
        """
        average_error = sum(self.errors) / self.size  # Compute mean absolute error
        rmsle = math.sqrt(sum(self.sles) / self.size)  # Compute RMSLE
        hits = sum(1 for color in self.colors if color == "green")  # Count green predictions
        hit_rate = hits / self.size * 100  # Compute hit percentage

        # Generate the plot with the calculated metrics
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hit_rate:.1f}%"
        self.chart(title)

    def run(self):
        """
        Execute the prediction test over the dataset.

        - Iterates through all selected data points.
        - Runs each datapoint through the predictor.
        - Generates a final report at the end.
        """
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)  # Process each data point
        self.report()  # Generate final summary

    @classmethod
    def test(cls, function, data):
        """
        Convenience method to quickly test a predictor function.

        :param function: The predictor function to evaluate.
        :param data: The dataset to use for testing.
        """
        cls(function, data).run()  # Instantiate and run the test
