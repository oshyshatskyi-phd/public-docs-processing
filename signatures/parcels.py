from typing import Literal
import dspy
from dspy import OutputField

class ParcelSignature(dspy.Signature):
    """Extract structured data about parcel properties from the text."""

    _CATEGORY_NAMES = [
        "Землі водного фонду",
        "Землі житлової та громадської забудови",
        "Землі історико-культурного призначення",
        "Землі лісогосподарського призначення",
        "Землі оздоровчого призначення",
        "Землі природно-заповідного та іншого природоохоронного призначення",
        "Землі промисловості, транспорту, зв'язку, енергетики, оборони та іншого призначення",
        "Землі рекреаційного призначення",
        "Землі сільськогосподарського призначення",
    ]

    _OWNERSHIP_NAMES = [
        "Державна власність",
        "Комунальна власність",
        "Приватна власність",
    ]

    id: int = OutputField(
        desc="Unique identifier of the documentation. Start with 0 and increment for each new documentation object. "
    )

    type: str = OutputField(
        default="parcel",
        desc="Type of the object, always 'parcel'. "
             "This field is used to distinguish between different types of objects in the output."
    )

    number: str | None = OutputField(
        pattern=r"^[0-9]+:[0-9]+:[0-9]+:[0-9]+$",
        desc="Cadastral number of the parcel. E.g. 7120382000:01:002:0021. "
             "Always match pattern ^[0-9]+:[0-9]+:[0-9]+:[0-9]+$. "
             "None if not specified or does not match the pattern",
        default=None,
    )
    area: float | None = OutputField(
        desc="Area of the parcel. None if not specified",
        default=None,
    )
    area_unit: Literal["ha", "m2"] | None = OutputField(
        desc="Unit of area measurement. None if not specified",
        default=None,
    )

    purpose_code: str | None = OutputField(
        desc="Purpose code of the parcel. E.g. 01.01, 10.04, 02.01. None if not specified",
        pattern=r"^[0-9]+.[0-9]+$",
        default=None,
    )
    category: Literal[*_CATEGORY_NAMES] | None = OutputField(
        desc="Category of the parcel. None if not specified",
        default=None,
    )

    ownership: Literal[*_OWNERSHIP_NAMES] | None = OutputField(
        desc="Ownership type of the parcel. "
             "Transfer to person makes it Приватна власність. "
             "Leasing does not change ownership. "
             "None if not specified",
        default=None,
    )
