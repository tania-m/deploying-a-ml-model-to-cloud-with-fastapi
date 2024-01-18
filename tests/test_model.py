# Tests for functions from model.py

from model import train_model, inference, MissingModelException

# Positive tests
def test_train_model():
    """
    Tests a model can be trained
    """
    pass

def compute_model_metrics():
    """
    Tests metrics computation
    """
    
def test_inference():
    """
    Tests inference function runs inference
    """
    pass

# Negative tests
def test_inference_error_handling():
    """
    Test no inference runs without model
    """
    try: 
        inference(None, None)
        # Should not go here, should have raised an exception
        assert False
    except MissingModelException:
        # Expected exception
        assert True
    except:
        # Unexpected exception
        assert False