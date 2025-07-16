import dspy
import pydantic
from signatures import (
    ParcelSignature,
    ParcelDocumentation,
    ParcelsListSignature,
    ParcelDocumentationReferences,
)


dspy.settings.configure(track_usage=True)

class Result(pydantic.BaseModel):
    """Entities extracted from the text."""
    entities: list
    usage: dict | None = None

    def set_lm_usage(self, usage: dict):
        """Set the LM usage information."""
        print("Setting LM usage:", usage)
        self.usage = usage

    def get_lm_usage(self) -> dict | None:
        """Get the LM usage information."""
        return self.usage


class ActionsAndEntities(dspy.Module):
    """Extract structured data about actions and entities from the text."""

    def __init__(self):
        super().__init__()

        self.identified_parcels = dspy.ChainOfThought(ParcelsListSignature)
        self.identified_documentation = dspy.ChainOfThought(ParcelDocumentationReferences)

    def forward(self, text: str) -> tuple:
        identified_parcels = self.identified_parcels(text=text)

        print(identified_parcels)

        parcels_usage = identified_parcels.get_lm_usage()

        for index, parcel in enumerate(identified_parcels.parcels):
            # reindex from 0
            parcel.id = index
        
        identified_documentation = self.identified_documentation(
            text=text,
            identified_parcels=identified_parcels.parcels
        )

        documentation_usage = identified_documentation.get_lm_usage()

        print(parcels_usage, documentation_usage)

        return Result(
            entities=[
                *identified_parcels.parcels, 
                *identified_documentation.referenced_documentation
            ]
        )
