"""The LDBC SNB Interactive v1 CSV layout, as the reference queries expect it.

The dataset ships a *storage* schema and the queries are written against a
*query* schema, and they are not the same thing. Three conversions are needed,
all of which the upstream Neo4j implementation also performs (in its
`convert-csvs.sh`); doing them at load time here keeps the query texts as close
to upstream as possible, which is the point of the exercise.

1. `place` and `organisation` are single files carrying a `type` column, but the
   queries match on `:City`, `:Country`, `:Continent`, `:Company` and
   `:University`. The type column becomes the label.
2. `:Comment` and `:Post` both need the `:Message` super-label — IC7 and IC8
   match `(:Message)` and would otherwise find nothing. FalkorDB supports
   multiple labels per node, so this is a plain second label rather than a
   duplicated node.
3. Self-referencing edge files have a duplicated header (`Person.id|Person.id`),
   which is ambiguous once parsed into a map: the second column shadows the
   first, both endpoints resolve to the same node, and every such edge becomes a
   self-loop. The header is rewritten to `FromX.id|ToX.id` before loading.

`birthdayMonth` / `birthdayDay` are derived here rather than in the query
because FalkorDB has no `datetime()`; see interactive-complex-10.cypher.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

#: Directory name inside the extracted tarball, parameterised by scale factor.
DATASET_DIR = "social_network-sf{sf}-CsvComposite-LongDateFormatter"

#: Tarball name on datasets.ldbcouncil.org.
DATASET_TARBALL = DATASET_DIR + ".tar.zst"

DATASET_BASE_URL = "https://datasets.ldbcouncil.org/snb-interactive-v1"

#: Scale factors this harness knows how to fetch. Both were verified to return
#: HTTP 200 with the sizes noted; larger SFs exist upstream but have not been
#: exercised here, and `load` at SF>1 has not been timed.
SCALE_FACTORS = {
    "0.1": 18_359_718,
    "1": 230_394_018,
}


class NodeFile(NamedTuple):
    """One node CSV and the pattern that loads it.

    label       primary label, or None when the label comes from `type_column`
    properties  Cypher expression per property, evaluated against `row`
    type_column column holding the label, for the polymorphic static files
    extra_labels additional labels every node in the file gets (`:Message`)
    """

    file: str
    label: str | None
    properties: dict[str, str]
    type_column: str | None = None
    extra_labels: tuple[str, ...] = ()


class EdgeFile(NamedTuple):
    """One edge CSV.

    from_id/to_id are the *rewritten* header names, so a self-referencing file
    names them distinctly. `from_label`/`to_label` are only used to pick the
    index for endpoint lookup; where the storage file is polymorphic the base
    label is used and the conversion above has already applied the subtype.
    """

    file: str
    type: str
    from_label: str
    to_label: str
    from_id: str
    to_id: str
    properties: Mapping[str, str] = MappingProxyType({})


#: Storage `type` values mapped to the labels the queries use. The datagen
#: emits these lowercase; the queries use CamelCase.
PLACE_LABELS = {"city": "City", "country": "Country", "continent": "Continent"}
ORGANISATION_LABELS = {"company": "Company", "university": "University"}


NODE_FILES = (
    NodeFile(
        file="static/place_0_0.csv",
        label="Place",
        type_column="type",
        properties={
            "id": "toInteger(row.id)",
            "name": "row.name",
            "url": "row.url",
        },
    ),
    NodeFile(
        file="static/organisation_0_0.csv",
        label="Organisation",
        type_column="type",
        properties={
            "id": "toInteger(row.id)",
            "name": "row.name",
            "url": "row.url",
        },
    ),
    NodeFile(
        file="static/tag_0_0.csv",
        label="Tag",
        properties={
            "id": "toInteger(row.id)",
            "name": "row.name",
            "url": "row.url",
        },
    ),
    NodeFile(
        file="static/tagclass_0_0.csv",
        label="TagClass",
        properties={
            "id": "toInteger(row.id)",
            "name": "row.name",
            "url": "row.url",
        },
    ),
    NodeFile(
        file="dynamic/person_0_0.csv",
        label="Person",
        properties={
            "id": "toInteger(row.id)",
            "firstName": "row.firstName",
            "lastName": "row.lastName",
            "gender": "row.gender",
            "birthday": "toInteger(row.birthday)",
            # IC10 filters on the month/day of the birthday and FalkorDB has no
            # datetime(). These two columns do not exist in the shipped CSV;
            # `dataset.prepare` appends them, deriving both from the epoch-millis
            # birthday. See interactive-complex-10.cypher.
            "birthdayMonth": "toInteger(row.birthdayMonth)",
            "birthdayDay": "toInteger(row.birthdayDay)",
            "creationDate": "toInteger(row.creationDate)",
            "locationIP": "row.locationIP",
            "browserUsed": "row.browserUsed",
            # IC1 returns these as lists; they are semicolon-joined in the CSV.
            "speaks": "CASE row.language WHEN '' THEN [] ELSE split(row.language, ';') END",
            "email": "CASE row.email WHEN '' THEN [] ELSE split(row.email, ';') END",
        },
    ),
    NodeFile(
        file="dynamic/forum_0_0.csv",
        label="Forum",
        properties={
            "id": "toInteger(row.id)",
            "title": "row.title",
            "creationDate": "toInteger(row.creationDate)",
        },
    ),
    NodeFile(
        file="dynamic/post_0_0.csv",
        label="Post",
        extra_labels=("Message",),
        properties={
            "id": "toInteger(row.id)",
            "imageFile": "row.imageFile",
            "creationDate": "toInteger(row.creationDate)",
            "locationIP": "row.locationIP",
            "browserUsed": "row.browserUsed",
            "language": "row.language",
            "content": "row.content",
            "length": "toInteger(row.length)",
        },
    ),
    NodeFile(
        file="dynamic/comment_0_0.csv",
        label="Comment",
        extra_labels=("Message",),
        properties={
            "id": "toInteger(row.id)",
            "creationDate": "toInteger(row.creationDate)",
            "locationIP": "row.locationIP",
            "browserUsed": "row.browserUsed",
            "content": "row.content",
            "length": "toInteger(row.length)",
        },
    ),
)


EDGE_FILES = (
    EdgeFile(
        file="static/organisation_isLocatedIn_place_0_0.csv",
        type="IS_LOCATED_IN",
        from_label="Organisation",
        to_label="Place",
        from_id="Organisation.id",
        to_id="Place.id",
    ),
    EdgeFile(
        file="static/place_isPartOf_place_0_0.csv",
        type="IS_PART_OF",
        from_label="Place",
        to_label="Place",
        from_id="FromPlace.id",
        to_id="ToPlace.id",
    ),
    EdgeFile(
        file="static/tag_hasType_tagclass_0_0.csv",
        type="HAS_TYPE",
        from_label="Tag",
        to_label="TagClass",
        from_id="Tag.id",
        to_id="TagClass.id",
    ),
    EdgeFile(
        file="static/tagclass_isSubclassOf_tagclass_0_0.csv",
        type="IS_SUBCLASS_OF",
        from_label="TagClass",
        to_label="TagClass",
        from_id="FromTagClass.id",
        to_id="ToTagClass.id",
    ),
    EdgeFile(
        file="dynamic/person_isLocatedIn_place_0_0.csv",
        type="IS_LOCATED_IN",
        from_label="Person",
        to_label="Place",
        from_id="Person.id",
        to_id="Place.id",
    ),
    EdgeFile(
        file="dynamic/person_knows_person_0_0.csv",
        type="KNOWS",
        from_label="Person",
        to_label="Person",
        from_id="FromPerson.id",
        to_id="ToPerson.id",
        properties={"creationDate": "toInteger(row.creationDate)"},
    ),
    EdgeFile(
        file="dynamic/person_hasInterest_tag_0_0.csv",
        type="HAS_INTEREST",
        from_label="Person",
        to_label="Tag",
        from_id="Person.id",
        to_id="Tag.id",
    ),
    EdgeFile(
        file="dynamic/person_studyAt_organisation_0_0.csv",
        type="STUDY_AT",
        from_label="Person",
        to_label="Organisation",
        from_id="Person.id",
        to_id="Organisation.id",
        properties={"classYear": "toInteger(row.classYear)"},
    ),
    EdgeFile(
        file="dynamic/person_workAt_organisation_0_0.csv",
        type="WORK_AT",
        from_label="Person",
        to_label="Organisation",
        from_id="Person.id",
        to_id="Organisation.id",
        properties={"workFrom": "toInteger(row.workFrom)"},
    ),
    EdgeFile(
        file="dynamic/person_likes_post_0_0.csv",
        type="LIKES",
        from_label="Person",
        to_label="Post",
        from_id="Person.id",
        to_id="Post.id",
        properties={"creationDate": "toInteger(row.creationDate)"},
    ),
    EdgeFile(
        file="dynamic/person_likes_comment_0_0.csv",
        type="LIKES",
        from_label="Person",
        to_label="Comment",
        from_id="Person.id",
        to_id="Comment.id",
        properties={"creationDate": "toInteger(row.creationDate)"},
    ),
    EdgeFile(
        file="dynamic/forum_hasModerator_person_0_0.csv",
        type="HAS_MODERATOR",
        from_label="Forum",
        to_label="Person",
        from_id="Forum.id",
        to_id="Person.id",
    ),
    EdgeFile(
        file="dynamic/forum_hasMember_person_0_0.csv",
        type="HAS_MEMBER",
        from_label="Forum",
        to_label="Person",
        from_id="Forum.id",
        to_id="Person.id",
        properties={"joinDate": "toInteger(row.joinDate)"},
    ),
    EdgeFile(
        file="dynamic/forum_hasTag_tag_0_0.csv",
        type="HAS_TAG",
        from_label="Forum",
        to_label="Tag",
        from_id="Forum.id",
        to_id="Tag.id",
    ),
    EdgeFile(
        file="dynamic/forum_containerOf_post_0_0.csv",
        type="CONTAINER_OF",
        from_label="Forum",
        to_label="Post",
        from_id="Forum.id",
        to_id="Post.id",
    ),
    EdgeFile(
        file="dynamic/post_hasCreator_person_0_0.csv",
        type="HAS_CREATOR",
        from_label="Post",
        to_label="Person",
        from_id="Post.id",
        to_id="Person.id",
    ),
    EdgeFile(
        file="dynamic/post_hasTag_tag_0_0.csv",
        type="HAS_TAG",
        from_label="Post",
        to_label="Tag",
        from_id="Post.id",
        to_id="Tag.id",
    ),
    EdgeFile(
        file="dynamic/post_isLocatedIn_place_0_0.csv",
        type="IS_LOCATED_IN",
        from_label="Post",
        to_label="Place",
        from_id="Post.id",
        to_id="Place.id",
    ),
    EdgeFile(
        file="dynamic/comment_hasCreator_person_0_0.csv",
        type="HAS_CREATOR",
        from_label="Comment",
        to_label="Person",
        from_id="Comment.id",
        to_id="Person.id",
    ),
    EdgeFile(
        file="dynamic/comment_hasTag_tag_0_0.csv",
        type="HAS_TAG",
        from_label="Comment",
        to_label="Tag",
        from_id="Comment.id",
        to_id="Tag.id",
    ),
    EdgeFile(
        file="dynamic/comment_isLocatedIn_place_0_0.csv",
        type="IS_LOCATED_IN",
        from_label="Comment",
        to_label="Place",
        from_id="Comment.id",
        to_id="Place.id",
    ),
    EdgeFile(
        file="dynamic/comment_replyOf_post_0_0.csv",
        type="REPLY_OF",
        from_label="Comment",
        to_label="Post",
        from_id="Comment.id",
        to_id="Post.id",
    ),
    EdgeFile(
        file="dynamic/comment_replyOf_comment_0_0.csv",
        type="REPLY_OF",
        from_label="Comment",
        to_label="Comment",
        from_id="FromComment.id",
        to_id="ToComment.id",
    ),
)


#: Upstream `indices.cypher`, translated. The unique constraints are expressed
#: as GRAPH.CONSTRAINT commands because FalkorDB rejects
#: `CREATE CONSTRAINT ... ASSERT`; each additionally *requires* a supporting
#: exact-match index to already exist, so the index list below is not merely an
#: optimisation.
UNIQUE_CONSTRAINTS = (
    "City",
    "Comment",
    "Country",
    "Forum",
    "Message",
    "Organisation",
    "Person",
    "Post",
    "Tag",
)

#: (label, property) exact-match indices. The first group backs the unique
#: constraints above; the second is upstream's own index list.
INDICES = (
    *((label, "id") for label in UNIQUE_CONSTRAINTS),
    ("Country", "name"),
    ("Message", "creationDate"),
    ("Person", "firstName"),
    ("Post", "creationDate"),
    ("Tag", "name"),
    ("TagClass", "name"),
    # Not upstream: the loader looks endpoints up by id on every edge file, and
    # these labels carry no unique constraint of their own.
    ("Place", "id"),
    ("TagClass", "id"),
    ("University", "id"),
    ("Company", "id"),
    ("Continent", "id"),
)
