from sam3_app.core.model import SAM3Model

# Global singleton
_model_instance = None

def get_model() -> SAM3Model:
    """Get the global SAM3Model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SAM3Model()
    return _model_instance
