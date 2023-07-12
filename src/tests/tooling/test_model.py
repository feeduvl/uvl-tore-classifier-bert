import pytest

from tooling.model import id_to_label
from tooling.model import Label_None_Pad
from tooling.model import label_to_id


@pytest.mark.parametrize(
    "label,id",
    [
        ("0", 0),
        ("Task", 1),
        ("Goals", 2),
        ("Domain_Data", 3),
        ("Activity", 4),
        ("Stakeholder", 5),
        ("System_Function", 6),
        ("Interaction", 7),
        ("Interaction_Data", 8),
        ("Workspace", 9),
        ("Software", 10),
        ("Internal_Action", 11),
        ("Internal_Data", 12),
        ("Domain_Level", 13),
        ("Interaction_Level", 14),
        ("System_Level", 15),
        ("_", -1),
    ],
)
def test_label_to_id(label: Label_None_Pad, id: int) -> None:
    assert label_to_id(label) == id
    assert id_to_label(id) == label
    assert id_to_label(label_to_id(label)) == label
    assert label_to_id(id_to_label(id)) == id
