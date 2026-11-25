import os
import subprocess

import common


def download_ldbc_data(filename):
    if not os.path.exists(f"data/{filename}"):
        url = f"https://repository.surfsara.nl/datasets/cwi/ldbc-snb-interactive-v1-datagen-v100/files/{filename}"
        subprocess.run(
            ["wget", "--no-check-certificate", url, "-O", filename],
            check=True,
            stdin=subprocess.PIPE,
            cwd="data",
        )

        # decompress the file
        subprocess.run(
            ["zstd", "--decompress", filename],
            check=True,
            stdin=subprocess.PIPE,
            cwd="data",
        )

        # extract the tar file
        subprocess.run(
            ["tar", "-xf", filename], check=True, stdin=subprocess.PIPE, cwd="data"
        )


def setup_module(module):
    files = [
        "social_network-sf0.1-CsvBasic-LongDateFormatter.tar.zst",
        "social_network-sf1-CsvBasic-LongDateFormatter.tar.zst",
    ]
    for file in files:
        download_ldbc_data(file)
    common.start_redis(moduleEnvs=["IMPORT_FOLDER", "data/"])


def teardown_module(module):
    common.shutdown_redis()


def setup_function(function):
    if common.g.name in common.client.list_graphs():
        common.g.delete()


base_path = "social_network-sf1-CsvBasic-LongDateFormatter"
node_files = [
    {
        "file": "static/organisation_0_0.csv",
        "label": "Organization",
        "properties": {
            "id": "toInteger(row.id)",
            "type": "row.type",
            "name": "row.name",
            "url": "row.url",
        },
    },
    {
        "file": "static/place_0_0.csv",
        "label": "Place",
        "properties": {
            "id": "toInteger(row.id)",
            "type": "row.type",
            "name": "row.name",
            "url": "row.url",
        },
    },
    {
        "file": "static/tag_0_0.csv",
        "label": "Tag",
        "properties": {
            "id": "toInteger(row.id)",
            "name": "row.name",
            "url": "row.url",
        },
    },
    {
        "file": "static/tagclass_0_0.csv",
        "label": "TagClass",
        "properties": {
            "id": "toInteger(row.id)",
            "name": "row.name",
            "url": "row.url",
        },
    },
    {
        "file": "dynamic/comment_0_0.csv",
        "label": "Comment",
        "properties": {
            "id": "toInteger(row.id)",
            "creationDate": "toInteger(row.creationDate)",
            "locationIP": "row.locationIP",
            "browserUsed": "row.browserUsed",
            "content": "row.content",
            "length": "toInteger(row.length)",
        },
    },
    {
        "file": "dynamic/forum_0_0.csv",
        "label": "Forum",
        "properties": {
            "id": "toInteger(row.id)",
            "title": "row.title",
            "creationDate": "toInteger(row.creationDate)",
        },
    },
    {
        "file": "dynamic/person_0_0.csv",
        "label": "Person",
        "properties": {
            "id": "toInteger(row.id)",
            "firstName": "row.firstName",
            "lastName": "row.lastName",
            "gender": "row.gender",
            "birthday": "toInteger(row.birthday)",
            "creationDate": "toInteger(row.creationDate)",
            "locationIP": "row.locationIP",
            "browserUsed": "row.browserUsed",
        },
    },
    {
        "file": "dynamic/post_0_0.csv",
        "label": "Post",
        "properties": {
            "id": "toInteger(row.id)",
            "imageFile": "row.imageFile",
            "creationDate": "toInteger(row.creationDate)",
            "locationIP": "row.locationIP",
            "browserUsed": "row.browserUsed",
            "language": "row.language",
            "content": "row.content",
            "length": "toInteger(row.length)",
        },
    },
]

edge_files = [
    {
        "file": "static/organisation_isLocatedIn_place_0_0.csv",
        "type": "IS_LOCATED_IN",
        "properties": {},
        "from_label": "Organization",
        "to_label": "Place",
        "from_id": "Organisation.id",
        "to_id": "Place.id",
    },
    {
        "file": "static/place_isPartOf_place_0_0.csv",
        "type": "IS_PART_OF",
        "properties": {},
        "from_label": "Place",
        "to_label": "Place",
        "from_id": "FromPlace.id",
        "to_id": "ToPlace.id",
    },
    {
        "file": "static/tag_hasType_tagclass_0_0.csv",
        "type": "HAS_TYPE",
        "properties": {},
        "from_label": "Tag",
        "to_label": "TagClass",
        "from_id": "Tag.id",
        "to_id": "TagClass.id",
    },
    {
        "file": "static/tagclass_isSubclassOf_tagclass_0_0.csv",
        "type": "IS_SUBCLASS_OF",
        "properties": {},
        "from_label": "TagClass",
        "to_label": "TagClass",
        "from_id": "FromTagClass.id",
        "to_id": "ToTagClass.id",
    },
    {
        "file": "dynamic/comment_hasCreator_person_0_0.csv",
        "type": "HAS_CREATOR",
        "properties": {},
        "from_label": "Comment",
        "to_label": "Person",
        "from_id": "Comment.id",
        "to_id": "Person.id",
    },
    {
        "file": "dynamic/comment_hasTag_tag_0_0.csv",
        "type": "HAS_TAG",
        "properties": {},
        "from_label": "Comment",
        "to_label": "Tag",
        "from_id": "Comment.id",
        "to_id": "Tag.id",
    },
    {
        "file": "dynamic/comment_isLocatedIn_place_0_0.csv",
        "type": "IS_LOCATED_IN",
        "properties": {},
        "from_label": "Comment",
        "to_label": "Place",
        "from_id": "Comment.id",
        "to_id": "Place.id",
    },
    {
        "file": "dynamic/comment_replyOf_comment_0_0.csv",
        "type": "REPLY_OF",
        "properties": {},
        "from_label": "Comment",
        "to_label": "Comment",
        "from_id": "FromComment.id",
        "to_id": "ToComment.id",
    },
    {
        "file": "dynamic/comment_replyOf_post_0_0.csv",
        "type": "REPLY_OF",
        "properties": {},
        "from_label": "Comment",
        "to_label": "Post",
        "from_id": "Comment.id",
        "to_id": "Post.id",
    },
    {
        "file": "dynamic/forum_containerOf_post_0_0.csv",
        "type": "CONTAINER_OF",
        "properties": {},
        "from_label": "Forum",
        "to_label": "Post",
        "from_id": "Forum.id",
        "to_id": "Post.id",
    },
    {
        "file": "dynamic/forum_hasMember_person_0_0.csv",
        "type": "HAS_MEMBER",
        "properties": {"joinDate": "toInteger(row.joinDate)"},
        "from_label": "Forum",
        "to_label": "Person",
        "from_id": "Forum.id",
        "to_id": "Person.id",
    },
    {
        "file": "dynamic/forum_hasModerator_person_0_0.csv",
        "type": "HAS_MODERATOR",
        "properties": {},
        "from_label": "Forum",
        "to_label": "Person",
        "from_id": "Forum.id",
        "to_id": "Person.id",
    },
    {
        "file": "dynamic/forum_hasTag_tag_0_0.csv",
        "type": "HAS_TAG",
        "properties": {},
        "from_label": "Forum",
        "to_label": "Tag",
        "from_id": "Forum.id",
        "to_id": "Tag.id",
    },
    {
        "file": "dynamic/person_hasInterest_tag_0_0.csv",
        "type": "HAS_INTEREST",
        "properties": {},
        "from_label": "Person",
        "to_label": "Tag",
        "from_id": "Person.id",
        "to_id": "Tag.id",
    },
    {
        "file": "dynamic/person_isLocatedIn_place_0_0.csv",
        "type": "IS_LOCATED_IN",
        "properties": {},
        "from_label": "Person",
        "to_label": "Place",
        "from_id": "Person.id",
        "to_id": "Place.id",
    },
    {
        "file": "dynamic/person_knows_person_0_0.csv",
        "type": "KNOWS",
        "properties": {"creationDate": "toInteger(row.creationDate)"},
        "from_label": "Person",
        "to_label": "Person",
        "from_id": "FromPerson.id",
        "to_id": "ToPerson.id",
    },
    {
        "file": "dynamic/person_likes_comment_0_0.csv",
        "type": "LIKES",
        "properties": {"creationDate": "toInteger(row.creationDate)"},
        "from_label": "Person",
        "to_label": "Comment",
        "from_id": "Person.id",
        "to_id": "Comment.id",
    },
    {
        "file": "dynamic/person_likes_post_0_0.csv",
        "type": "LIKES",
        "properties": {"creationDate": "toInteger(row.creationDate)"},
        "from_label": "Person",
        "to_label": "Post",
        "from_id": "Person.id",
        "to_id": "Post.id",
    },
    {
        "file": "dynamic/person_studyAt_organisation_0_0.csv",
        "type": "STUDY_AT",
        "properties": {"classYear": "toInteger(row.classYear)"},
        "from_label": "Person",
        "to_label": "Organization",
        "from_id": "Person.id",
        "to_id": "Organisation.id",
    },
    {
        "file": "dynamic/person_workAt_organisation_0_0.csv",
        "type": "WORK_AT",
        "properties": {"workFrom": "toInteger(row.workFrom)"},
        "from_label": "Person",
        "to_label": "Organization",
        "from_id": "Person.id",
        "to_id": "Organisation.id",
    },
    {
        "file": "dynamic/post_hasCreator_person_0_0.csv",
        "type": "HAS_CREATOR",
        "properties": {},
        "from_label": "Post",
        "to_label": "Person",
        "from_id": "Post.id",
        "to_id": "Person.id",
    },
    {
        "file": "dynamic/post_hasTag_tag_0_0.csv",
        "type": "HAS_TAG",
        "properties": {},
        "from_label": "Post",
        "to_label": "Tag",
        "from_id": "Post.id",
        "to_id": "Tag.id",
    },
    {
        "file": "dynamic/post_isLocatedIn_place_0_0.csv",
        "type": "IS_LOCATED_IN",
        "properties": {},
        "from_label": "Post",
        "to_label": "Place",
        "from_id": "Post.id",
        "to_id": "Place.id",
    },
]


def test_load_csv():
    total_time = 0
    for file in node_files:
        common.g.query(f"CREATE INDEX FOR (n:{file['label']}) ON (n.id)")
        query = f"""
            LOAD CSV WITH HEADERS DELIMITER '|' FROM $file AS row
            RETURN count(row)
            """
        res = common.g.query(
            query,
            {
                "file": f"file://{base_path}/{file['file']}",
            },
        )
        expected = res.result_set[0][0]
        properties_str = ", ".join(
            f"{key}: {value}" for key, value in file["properties"].items()
        )
        query = f"""
            LOAD CSV WITH HEADERS DELIMITER '|' FROM $file AS row
            CREATE (:{file['label']} {{{properties_str}}})
            """
        res = common.g.query(
            query,
            {
                "file": f"file://{base_path}/{file['file']}",
            },
        )
        assert res.nodes_created == expected
        total_time += res.run_time_ms
        print(res.run_time_ms)

    for file in edge_files:
        if file["from_label"] == file["to_label"]:
            with open(f"data/{base_path}/{file['file']}") as f:
                line = f.readline().split("|")
                line[0] = file["from_id"]
                line[1] = file["to_id"]
                line = "|".join(line)
                line = line.replace("\n", "")
                subprocess.run(["sed", "-i", "", f"1s/.*/{line}/", f"data/{base_path}/{file['file']}"], check=True)
        query = f"""
            LOAD CSV WITH HEADERS DELIMITER '|' FROM $file AS row
            RETURN count(row)
            """
        res = common.g.query(
            query,
            {
                "file": f"file://{base_path}/{file['file']}",
            },
        )
        expected = res.result_set[0][0]
        properties_str = ", ".join(
            f"{key}: {value}" for key, value in file["properties"].items()
        )
        query = f"""
            LOAD CSV WITH HEADERS DELIMITER '|' FROM $file AS row
            MATCH (f:{file['from_label']} {{id: toInteger(row.`{file['from_id']}`)}})
            WITH f, row
            MATCH (t:{file['to_label']} {{id: toInteger(row.`{file['to_id']}`)}})
            CREATE (f)-[r:{file['type']} {{{properties_str}}}]->(t)
            """
        res = common.g.query(
            query,
            {
                "file": f"file://{base_path}/{file['file']}",
            },
        )
        assert res.relationships_created == expected
        total_time += res.run_time_ms
        print(res.run_time_ms)

    print(f"Total time for loading CSV files: {total_time} ms")
