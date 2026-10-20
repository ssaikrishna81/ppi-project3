"""Simple semantic memory for observations."""
class SemanticMemory:
    def __init__(self):
        self.objects = {}

    def update(self, key: str, location: str, confidence: float):
        self.objects[key] = {
            "location": location,
            "confidence": float(confidence),
        }

    def query(self, key: str):
        return self.objects.get(key)

    def as_dict(self):
        return dict(self.objects)
