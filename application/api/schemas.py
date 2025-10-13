from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Common Schemas ---
class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    model_status: Optional[str] = Field(None, example="loaded")

# --- Prediction Schemas ---
# Example for a model that takes numerical features and returns a class and probability
class TabularInputFeatures(BaseModel):
    feature1: float = Field(..., example=10.5)
    feature2: float = Field(..., example=-2.3)
    # Add all your model's features here with appropriate types and examples

class TextGenerationInput(BaseModel):
    prompt: str = Field(..., example="Translate the following English text to French: 'Hello, world!'")
    max_tokens: Optional[int] = Field(50, example=50)

class ClassificationPrediction(BaseModel):
    predicted_label: str = Field(..., example="Positive")
    confidence_score: Optional[float] = Field(None, example=0.95)
    # You can add more fields like class probabilities if needed
    # class_probabilities: Optional[Dict[str, float]] = Field(None, example={"Positive": 0.95, "Negative": 0.05})

class TextGenerationOutput(BaseModel):
    generated_text: str = Field(..., example="Bonjour le monde!")

# --- Request Body Schemas (used by endpoints) ---
class PredictTabularRequest(TabularInputFeatures): # Inherits all features
    pass

class PredictTextGenerationRequest(TextGenerationInput):
    pass

# --- Response Body Schemas (used by endpoints) ---
class PredictTabularResponse(ClassificationPrediction): # Inherits output fields
    request_id: Optional[str] = Field(None, example="abc-123") # Example of adding metadata

class PredictTextGenerationResponse(TextGenerationOutput):
    request_id: Optional[str] = Field(None, example="xyz-789")

# If you have batch prediction, you might define list types:
# class BatchPredictTabularRequest(BaseModel):
#     instances: List[TabularInputFeatures]

# class BatchPredictTabularResponse(BaseModel):
#     predictions: List[ClassificationPrediction]


if __name__ == '__main__':
    # Example Usage:
    # input_data = TabularInputFeatures(feature1=1.0, feature2=2.0)
    # print(input_data.json(indent=2))

    # output_data = ClassificationPrediction(predicted_label="ClassA", confidence_score=0.88)
    # print(output_data.json(indent=2))
    print("Pydantic schemas for API request/response validation placeholder.")

