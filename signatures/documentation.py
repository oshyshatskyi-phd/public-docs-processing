from typing import Literal, List
import dspy
from dspy import OutputField

class ParcelDocumentation(dspy.Signature):
    """
    Identify land management documentation that might be referenced in the text.

    Refer them to identified_parcels list, if possible.
    Here is an example of how to link parcels and documents with this signature:

    {
        "number": "100000001:20:100:1000",
        "area": null,
        "area_unit": "ha",
        "purpose_code": null,
        "category": "Землі житлової та громадської забудови",
        "ownership": "Приватна власність",
        "id": 0,
        "type": "parcel"
    },
    {
        "documentation_type": "BOUNDARY_ESTABLISHMENT_DOCUMENTATION",
        "involved_parcels": [
            0 # This is the ID of the parcel above
        ],
        "id": 0,
        "type": "documentation"
    }
    """
    id: int = OutputField(
        desc="Unique identifier of the documentation. Start with 0 and increment for each new documentation object. "
    )

    type: str = OutputField(default="documentation", desc="Type of the object, always 'documentation'")

    documentation_type: Literal[
        'LAND_USE_AND_JUSTIFICATION_SCHEME',
        'COMMUNITY_BOUNDARY_ESTABLISHMENT_PROJECT',
        'ADMIN_UNIT_BOUNDARY_PROJECT',
        'URBAN_PLANNING_DOCUMENTATION',
        'ECOLOGICAL_AND_RESTRICTED_USE_PROJECT',
        'AGRICULTURAL_ENTERPRISE_PRIVATIZATION_PROJECT',
        'LAND_PLOT_ALLOCATION_PROJECT',
        'URBAN_TERRITORY_PLANNING_PROJECT',
        'CROP_ROTATION_AND_LAND_USE_PROJECT',
        'SETTLEMENT_TERRITORY_PLANNING_PROJECT',
        'LAND_SHARE_ORGANIZATION_PROJECT',
        'WORKING_PROJECT',
        'BOUNDARY_ESTABLISHMENT_DOCUMENTATION',
        'PARTIAL_RIGHTS_BOUNDARY_DOCUMENTATION',
        'PLOT_DIVISION_AND_MERGE_DOCUMENTATION',
        'LAND_INVENTORY_DOCUMENTATION',
        'CONSERVATION_RESERVATION_DOCUMENTATION',
        'CULTURAL_HERITAGE_BOUNDARY_DOCUMENTATION',
        'OTHER_DOCUMENTATION',
    ] = OutputField(desc="""Detailed description of the values:
LAND_USE_AND_JUSTIFICATION_SCHEME = Схема землеустрою і техніко-економічні обґрунтування використання та охорони земель адміністративно-територіальних одиниць.
COMMUNITY_BOUNDARY_ESTABLISHMENT_PROJECT = Проект землеустрою щодо встановлення меж територій територіальних громад.
ADMIN_UNIT_BOUNDARY_PROJECT = Проект землеустрою щодо встановлення (зміни) меж адміністративно-територіальних одиниць.
URBAN_PLANNING_DOCUMENTATION  Містобудівна документація, яка одночасно є документацією із землеустрою (комплексні плани, генплани, детальні плани).
ECOLOGICAL_AND_RESTRICTED_USE_PROJECT  Проект щодо організації та встановлення меж територій природоохоронного, рекреаційного, історико-культурного, лісового, водного призначення.
AGRICULTURAL_ENTERPRISE_PRIVATIZATION_PROJECT  Проект приватизації земель державних і комунальних сільськогосподарських підприємств, установ та організацій.
LAND_PLOT_ALLOCATION_PROJECT  Проект землеустрою щодо відведення земельних ділянок.
URBAN_TERRITORY_PLANNING_PROJECT  Проект впорядкування території для містобудівних потреб.
CROP_ROTATION_AND_LAND_USE_PROJECT  Проект, що забезпечують еколого-економічне обґрунтування сівозміни та впорядкування угідь.
SETTLEMENT_TERRITORY_PLANNING_PROJECT = Проект щодо впорядкування території населених пунктів.
LAND_SHARE_ORGANIZATION_PROJECT = Проект організації території земельних часток (паїв).
WORKING_PROJECT = Робочий проект землеустрою.
BOUNDARY_ESTABLISHMENT_DOCUMENTATION = Технічна документація щодо встановлення (відновлення) меж земельної ділянки в натурі (на місцевості).
PARTIAL_RIGHTS_BOUNDARY_DOCUMENTATION = Технічна документація щодо встановлення меж частини земельної ділянки, на яку поширюються права суборенди, сервітуту.
PLOT_DIVISION_AND_MERGE_DOCUMENTATION = Технічна документація щодо поділу та об'єднання земельних ділянок.
LAND_INVENTORY_DOCUMENTATION = Технічна документація щодо інвентаризації земель.
CONSERVATION_RESERVATION_DOCUMENTATION = Технічна документація щодо резервування цінних для заповідання територій та об'єктів.
CULTURAL_HERITAGE_BOUNDARY_DOCUMENTATION = Технічна документація щодо встановлення меж режимоутворюючих об'єктів культурної спадщини.
OTHER_DOCUMENTATION = Технічна документація яка не підпадає під попередні категорії.
""")
    
    involved_parcels: List[int] = OutputField(
        desc='List of parcel identifiers that this documentation references. '
             'Use numeric parcel IDs from identified_parcels list. '
             "Otherwise, do not include any parcel identifiers.")
