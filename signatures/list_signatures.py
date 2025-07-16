from typing import List
import dspy
from dspy import InputField, OutputField

from .parcels import ParcelSignature
from .documentation import ParcelDocumentation

class ParcelsListSignature(dspy.Signature):
    """Extract structured data about parcels from the text."""
    text: str = InputField(desc="The text to extract data from")
    parcels: List[ParcelSignature] = OutputField(
        desc="List of parcels referenced in the text.",
        default_factory=list,
    )


class ParcelDocumentationReferences(dspy.Signature):
    """
    Extract list documentation that might be referenced in the text. 
    
    The only documentation that you must extract is that one that 100% matches
    one of the possible values of documentation_type.
    """
    
    text = InputField(
        desc="A snippet of text from a legal document where parcel events might be described."
    )

    identified_parcels: List[ParcelSignature] = InputField(
        desc="Optional. A list of parcel objects, previously identified. "
             "Helps link events to specific parcel IDs.",
        default_factory=list,
    )

    referenced_documentation: List[ParcelDocumentation] = OutputField(
        desc="List of documentation that might be referenced in the text",
        default_factory=list,
    )
