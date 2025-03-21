# %%
import json
import logging
import os
import torch
from ts.torch_handler.base_handler import BaseHandler
from model import NetworkWithCategoryEmbedding


logger = logging.getLogger(__name__)


class PorterModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        logger.info("GAUTAM Initializing model handler... INIT")

    def initialize(self, context):
        """Initialize model and other artifacts."""
        logger.info("GAUTAM Initializing model handler...")
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load model
        model_file = self.manifest["model"].get(
            "modelFile") or self.manifest["model"].get("serializedFile")
        if model_file is None:
            raise RuntimeError("Model file path is missing in manifest.")

        model_path = os.path.join(model_dir, model_file)
        logger.info("GAUTAM Model path:", model_path)

        # Initialize model
        self.device = "cpu"
        # Initialize with same params as your best_model
        self.model = torch.load(
            model_path, map_location=self.device,  weights_only=False)
        self.model.to(self.device)
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        """Process input data before model inference."""
        preprocessed_data = []
        logger.info(f"GAUTAM Received input data: {data}")
        for row in data:
            # Get the input data
            input_text = row.get("data") or row.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            # Parse JSON if the input is a JSON string
            json_data = json.loads(input_text)

            # Extract features and category
            X = torch.tensor(json_data["features"],
                             dtype=torch.float32).to(self.device)
            category = torch.tensor(
                json_data["category"], dtype=torch.long).to(self.device)

            preprocessed_data.append((X, category))

        return preprocessed_data

    def inference(self, data):
        """Run model inference on preprocessed data."""
        results = []
        logger.info(f"GAUTAM Running inference on data: {data}")

        for X, category in data:
            # Add batch dimension if needed
            if len(X.shape) == 1:
                X = X.unsqueeze(0)
                category = category.unsqueeze(0)

            with torch.no_grad():
                prediction = self.model(X, category)
                results.append(prediction)

        return results

    def postprocess(self, inference_output):
        """Process inference output to final results."""
        results = []

        for output in inference_output:
            result = output.cpu().numpy().tolist()
            results.append(result)

        return results


# # Add to the bottom of handler.py
# if __name__ == "__main__":
#     # Mock context
#     import os
#     from collections import namedtuple
#     Context = namedtuple('Context', ['manifest', 'system_properties'])
#     manifest = {
#         'model': {
#             'serializedFile': './model/porter_model_full.pth'
#         }
#     }
#     system_properties = {
#         'model_dir': './'
#     }
#     context = Context(manifest=manifest, system_properties=system_properties)

#     # Initialize handler
#     handler = PorterModelHandler()
#     handler.initialize(context)

#     # Create test input
#     test_input = [{
#         "data": json.dumps({
#             "features": [0.7274, 0.2505, 0.2039, -0.1874, -0.9148, -1.0736, -0.9917, -0.9009,
#                          -0.9192, -1.2392, 1.0207, 0.9981, -0.7344, -0.7538, -0.6802, -0.7363,
#                          -0.6794, -0.6097, -0.5396, -0.5624, -0.5178, -0.5654, -0.5224, -0.4787,
#                          -0.5506, -0.5190, -0.4762, -0.4351, -0.7250, -1.2671, 1.4097, 0.2309,
#                          -0.9985, 0.8317, 1.6118, -0.3703, -0.6053, -0.3280, -0.5511, -0.0558,
#                          -0.0102, -0.5030, -0.6694, 2.7626, -0.6041, -0.3364, -0.0615],
#             "category": 17
#         })
#     }]

#     # Process the input
#     preprocessed = handler.preprocess(test_input)
#     predictions = handler.inference(preprocessed)
#     result = handler.postprocess(predictions)

#     print("Test result:", result)
