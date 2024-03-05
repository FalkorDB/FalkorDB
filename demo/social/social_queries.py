import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from demo import QueryInfo


graph_entities = QueryInfo(
    query="""MATCH (e) RETURN e.name, LABELS(e) as label ORDER BY label, e.name""",
    description='Returns each node in the graph, specifing node label.',
    expected_result=[['Netherlands',['country']],
                     ['Andora',['country']],
                     ['Canada',['country']],
                     ['China',['country']],
                     ['Germany',['country']],
                     ['Greece',['country']],
                     ['Italy',['country']],
                     ['Japan',['country']],
                     ['Kazakhstan',['country']],
                     ['Prague',['country']],
                     ['Russia',['country']],
                     ['Thailand',['country']],
                     ['USA',['country']],
                     ['Ailon Velger',['person']],
                     ['Alon Fital',['person']],
                     ['Boaz Arad',['person']],
                     ['Gal Derriere',['person']],
                     ['Jane Chernomorin',['person']],
                     ['Lucy Yanfital',['person']],
                     ['Mor Yesharim',['person']],
                     ['Noam Nativ',['person']],
                     ['Omri Traub',['person']],
                     ['Ori Laslo',['person']],
                     ['Roi Lipman',['person']],
                     ['Shelly Laslo Rooz',['person']],
                     ['Tal Doron',['person']],
                     ['Valerie Abigail Arad',['person']]]
)

relation_type_counts = QueryInfo(
    query="""MATCH ()-[e]->() RETURN TYPE(e) as relation_type, COUNT(e) as num_relations ORDER BY relation_type, num_relations""",
    description='Returns each relation type in the graph and its count.',
    expected_result=[['friend', 13],
                     ['visited', 43]]
)

subset_of_people = QueryInfo(
    query="""MATCH (p:person) RETURN p.name ORDER BY p.name SKIP 3 LIMIT 5""",
    description='Get a subset of people.',
    expected_result=[["Gal Derriere"],
                    ["Jane Chernomorin"],
                    ["Lucy Yanfital"],
                    ["Mor Yesharim"],
                    ["Noam Nativ"]]
)

my_friends_query = QueryInfo(
    query="""MATCH (ME:person {name:"Roi Lipman"})-[:friend]->(f:person) 
             RETURN f.name""",
    description='My friends?',
    expected_result=[['Tal Doron'],
                     ['Omri Traub'],
                     ['Boaz Arad'],
                     ['Ori Laslo'],
                     ['Ailon Velger'],
                     ['Alon Fital']]
)

friends_of_friends_query = QueryInfo(
    query="""MATCH (ME:person {name:"Roi Lipman"})-[:friend]->(:person)-[:friend]->(fof:person) 
             RETURN fof.name""",
    description='Friends of friends?',
    expected_result=[['Valerie Abigail Arad'],
                     ['Shelly Laslo Rooz'],
                     ['Noam Nativ'],
                     ['Jane Chernomorin'],
                     ['Mor Yesharim'],
                     ['Gal Derriere'],
                     ['Lucy Yanfital']]
)

friends_of_friends_single_and_over_30_query = QueryInfo(
    query="""MATCH (ME:person {name:"Roi Lipman"})-[:friend]->(:person)-[:friend]->(fof:person {status:"single"})
             WHERE fof.age > 30
             RETURN fof.name, fof.age, fof.gender, fof.status""",
    description='Friends of friends who are single and over 30?',
    expected_result=[['Noam Nativ', 34, 'male', 'single']]
)

friends_of_friends_visited_netherlands_and_single_query = QueryInfo(
    query="""MATCH (ME:person {name:"Roi Lipman"})-[:friend]->(:person)-[:friend]->
             (fof:person {status:"single"})-[:visited]->(:country {name:"Netherlands"})
             RETURN fof.name ORDER BY fof.name""",
    description='Friends of friends who visited Netherlands and are single?',
    expected_result=[['Gal Derriere'],
                     ['Noam Nativ']]
)

friends_visited_same_places_as_me_query = QueryInfo(
    query="""MATCH (ME:person {name:"Roi Lipman"})-[:visited]->(c:country)<-[:visited]-(f:person)<-[:friend]-(ME)
             RETURN f.name, c.name ORDER BY f.name, c.name""",
    description='Friends who have been to places I have visited?',
    expected_result=[['Tal Doron', 'Japan'],
                     ['Alon Fital', 'Prague'],
                     ['Tal Doron', 'USA'],
                     ['Omri Traub', 'USA'],
                     ['Boaz Arad', 'USA'],
                     ['Ori Laslo', 'USA'],
                     ['Alon Fital', 'USA']]
)

countries_visited_by_roi_tal_boaz = QueryInfo(
    query="""MATCH (A:person {name:"Roi Lipman"})-[:visited]->(X:country),
                   (B:person {name:"Tal Doron"})-[:visited]->(X),
                   (C:person {name:"Boaz Arad"})-[:visited]->(X)
            RETURN X.name""",
    description='Countries visited by Roi, Tal and Boaz.',
    expected_result=[['USA']]
)

friends_older_than_me_query = QueryInfo(
    query="""MATCH (ME:person {name:"Roi Lipman"})-[:friend]->(f:person)
             WHERE f.age > ME.age
             RETURN f.name, f.age""",
    description='Friends who are older than me?',
    expected_result=[['Omri Traub', 33]]
)

friends_age_difference_query = QueryInfo(
    query="""MATCH (ME:person {name:"Roi Lipman"})-[:friend]->(f:person)
             RETURN f.name, abs(ME.age - f.age) AS age_diff
             ORDER BY age_diff desc""",
    description='Age difference between me and each of my friends.',
    expected_result=[['Boaz Arad', 1],
                     ['Omri Traub', 1],
                     ['Ailon Velger', 0],
                     ['Tal Doron', 0],
                     ['Ori Laslo', 0],
                     ['Alon Fital', 0]]
)

friends_who_are_older_than_average = QueryInfo(
    query="""MATCH (p:person)
             WITH avg(p.age) AS average_age 
             MATCH(:person)-[:friend]->(f:person) 
             WHERE f.age > average_age 
             RETURN f.name, f.age, round(f.age - average_age) AS age_diff 
             ORDER BY age_diff DESC, f.name DESC
             LIMIT 4""",
    description='Friends who are older then the average age.',
    expected_result=[['Noam Nativ', 34, 3.0],
                     ['Omri Traub', 33, 2.0],
                     ['Tal Doron', 32, 1.0],
                     ['Ori Laslo', 32, 1.0]]
)

how_many_countries_each_friend_visited_query = QueryInfo(
    query="""MATCH (ME:person {name:"Roi Lipman"})-[:friend]->(friend:person)-[:visited]->(c:country)
             RETURN friend.name, count(c.name) AS countriesVisited
             ORDER BY countriesVisited DESC
             LIMIT 10""",
    description='Count for each friend how many countires he or she been to?',
    expected_result=[['Alon Fital', 3],
                     ['Omri Traub', 3],
                     ['Tal Doron', 3],
                     ['Ori Laslo', 3],
                     ['Boaz Arad', 2]]
)

happy_birthday_query = QueryInfo(
    query = """MATCH (:person {name:"Roi Lipman"})-[:friend]->(f:person)
               SET f.age = f.age + 1
               RETURN f.name, f.age order by f.name, f.age""",
    description='Update friends age.',
    expected_result=[['Ailon Velger', 33],
                     ['Alon Fital',   33],
                     ['Boaz Arad',    32],
                     ['Omri Traub',   34],
                     ['Ori Laslo',    33],
                     ['Tal Doron',    33]]
)

friends_age_statistics_query = QueryInfo(
    query="""MATCH (ME:person {name:"Roi Lipman"})-[:friend]->(f:person)
             RETURN ME.name, count(f.name), sum(f.age), avg(f.age), min(f.age), max(f.age)""",
    description='Friends age statistics.',
    expected_result=[['Roi Lipman', 6, 198.0, 33.0, 32, 34]]
)

visit_purpose_of_each_country_i_visited_query = QueryInfo(
    query="""MATCH (ME:person {name:"Roi Lipman"})-[v:visited]->(c:country) 
             RETURN c.name, v.purpose""",
    description='For each country i have been to, what was the visit purpose?',
    expected_result=[['Japan', 'pleasure'],
                     ['Prague', 'pleasure'],
                     ['Prague', 'business'],
                     ['USA', 'business']]
)

who_was_on_business_trip_query = QueryInfo(
    query="""MATCH (p:person)-[v:visited {purpose:"business"}]->(c:country)
             RETURN p.name, v.purpose, toUpper(c.name) ORDER BY p.name, c.name""",
    description='Find out who went on a business trip?',
    expected_result=[['Boaz Arad', 'business','NETHERLANDS'],
                     ['Boaz Arad', 'business','USA'],
                     ['Ori Laslo', 'business', 'CHINA'],
                     ['Ori Laslo', 'business', 'USA'],
                     ['Jane Chernomorin', 'business', 'USA'],
                     ['Alon Fital', 'business', 'USA'],
                     ['Alon Fital', 'business', 'PRAGUE'],
                     ['Mor Yesharim', 'business', 'GERMANY'],
                     ['Gal Derriere', 'business', 'NETHERLANDS'],
                     ['Lucy Yanfital', 'business', 'USA'],
                     ['Roi Lipman', 'business', 'USA'],
                     ['Roi Lipman', 'business', 'PRAGUE'],
                     ['Tal Doron', 'business', 'USA'],
                     ['Tal Doron', 'business', 'JAPAN']]
)

number_of_vacations_per_person_query = QueryInfo(
    query="""MATCH (p:person)-[v:visited {purpose:"pleasure"}]->(c:country)
             RETURN p.name, count(v.purpose) AS vacations
             ORDER BY COUNT(v.purpose) DESC, p.name DESC
             LIMIT 6""",
    description='Count number of vacations per person?',
    expected_result=[['Shelly Laslo Rooz', 3],
                     ['Omri Traub', 3],
                     ['Noam Nativ', 3],
                     ['Lucy Yanfital', 3],
                     ['Jane Chernomorin', 3],
                     ['Alon Fital', 3]]
)

all_reachable_friends_query = QueryInfo(
    query="""MATCH (a:person {name:'Roi Lipman'})-[:friend*]->(b:person)
             RETURN b.name
             ORDER BY b.name""",
    description='Find all reachable friends',
    expected_result=[['Ailon Velger'],
                     ['Alon Fital'],
                     ['Boaz Arad'],
                     ['Gal Derriere'],
                     ['Jane Chernomorin'],
                     ['Lucy Yanfital'],
                     ['Mor Yesharim'],
                     ['Noam Nativ'],
                     ['Omri Traub'],
                     ['Ori Laslo'],
                     ['Shelly Laslo Rooz'],
                     ['Tal Doron'],
                     ['Valerie Abigail Arad']]
)

all_reachable_countries_query = QueryInfo(
    query="""MATCH (a:person {name:'Roi Lipman'})-[*]->(c:country)
             RETURN c.name, count(c.name) AS NumPathsToCountry
             ORDER BY NumPathsToCountry, c.name DESC""",
    description='Find all reachable countries',
    expected_result=[['USA', 14],
                     ['Netherlands', 6],
                     ['Prague', 5],
                     ['Greece', 4],
                     ['Japan', 2],
                     ['Germany', 2],
                     ['China', 2],
                     ['Canada', 2],
                     ['Andora', 2],
                     ['Thailand', 1],
                     ['Russia', 1],
                     ['Kazakhstan', 1],
                     ['Italy', 1]]
)

reachable_countries_or_people_query = QueryInfo(
    query="""MATCH (s:person {name:'Roi Lipman'})-[e:friend|:visited]->(t)
             RETURN s.name,TYPE(e),t.name
             ORDER BY t.name""",
    description='Every person or country one hop away from source node',
    expected_result=[["Roi Lipman", "friend", "Ailon Velger"],
                     ["Roi Lipman", "friend", "Alon Fital"],
                     ["Roi Lipman", "friend", "Boaz Arad"],
                     ["Roi Lipman", "visited", "Japan"],
                     ["Roi Lipman", "friend", "Omri Traub"],
                     ["Roi Lipman", "friend", "Ori Laslo"],
                     ["Roi Lipman", "visited", "Prague"],
                     ["Roi Lipman", "visited", "Prague"],
                     ["Roi Lipman", "friend", "Tal Doron"],
                     ["Roi Lipman", "visited", "USA"]]
)

all_reachable_countries_or_people_query = QueryInfo(
    query="""MATCH (a:person {name:'Roi Lipman'})-[:friend|:visited*]->(e) RETURN e.name, count(e.name) AS NumPathsToEntity ORDER BY NumPathsToEntity, e.name DESC""",
    description='Every reachable person or country from source node',
    expected_result=[['USA', 14],
                     ['Netherlands', 6],
                     ['Prague', 5],
                     ['Greece', 4],
                     ['Japan', 2],
                     ['Germany', 2],
                     ['China', 2],
                     ['Canada', 2],
                     ['Andora', 2],
                     ['Valerie Abigail Arad', 1],
                     ['Thailand', 1],
                     ['Tal Doron', 1],
                     ['Shelly Laslo Rooz', 1],
                     ['Russia', 1],
                     ['Ori Laslo', 1],
                     ['Omri Traub', 1],
                     ['Noam Nativ', 1],
                     ['Mor Yesharim', 1],
                     ['Lucy Yanfital', 1],
                     ['Kazakhstan', 1],
                     ['Jane Chernomorin', 1],
                     ['Italy', 1],
                     ['Gal Derriere', 1],
                     ['Boaz Arad', 1],
                     ['Alon Fital', 1],
                     ['Ailon Velger', 1]]
)

all_reachable_entities_query = QueryInfo(
    query="""MATCH (a:person {name:'Roi Lipman'})-[*]->(e)
             RETURN e.name, count(e.name) AS NumPathsToEntity
             ORDER BY NumPathsToEntity DESC""",
    description='Find all reachable entities',
    expected_result=[['USA', 14],
                     ['Netherlands', 6],
                     ['Prague', 5],
                     ['Greece', 4],
                     ['Andora', 2],
                     ['Japan', 2],
                     ['Germany', 2],
                     ['Canada', 2],
                     ['China', 2],
                     ['Ailon Velger', 1],
                     ['Alon Fital', 1],
                     ['Gal Derriere', 1],
                     ['Jane Chernomorin', 1],
                     ['Omri Traub', 1],
                     ['Boaz Arad', 1],
                     ['Noam Nativ', 1],
                     ['Shelly Laslo Rooz', 1],
                     ['Russia', 1],
                     ['Valerie Abigail Arad', 1],
                     ['Mor Yesharim', 1],
                     ['Italy', 1],
                     ['Tal Doron', 1],
                     ['Thailand', 1],
                     ['Kazakhstan', 1],
                     ['Lucy Yanfital', 1],
                     ['Ori Laslo', 1]]
)

all_reachable_people_min_2_hops_query = QueryInfo(
    query="""MATCH (ME:person {name:'Roi Lipman'})-[*2..]->(e:person)
             RETURN e.name
             ORDER BY e.name""",
    description='Find all reachable people at least 2 hops away from me',
    expected_result=[['Gal Derriere'],
                     ['Jane Chernomorin'],
                     ['Lucy Yanfital'],
                     ['Mor Yesharim'],
                     ['Noam Nativ'],
                     ['Shelly Laslo Rooz'],
                     ['Valerie Abigail Arad']]
)

all_paths_leads_to_greece_query = QueryInfo(
    query="""MATCH (a)-[*]->(e:country {name:'Greece'})
             RETURN count(a.name) AS NumPathsToGreece""",
    description='Number of paths leading to Greece',
    expected_result=[[10]]
)

number_of_paths_to_places_visited = QueryInfo(
    query="""MATCH (ME:person {name:'Roi Lipman'})-[:visited]->(c:country)<-[*]-(ME)
             RETURN c.name, count(c)
             ORDER BY c.name""",
    description='Count number of paths to places I have visited',
    expected_result=[['Japan', 2],
                     ['Prague', 5],
                     ['USA', 14]]
)

pagerank_friends = QueryInfo(
    query="""CALL algo.pageRank('person', 'friend') YIELD node, score
             RETURN node.name
             ORDER BY score DESC""",
    description='Pagerank friends',
    expected_result=[['Valerie Abigail Arad'],
                     ['Shelly Laslo Rooz'],
                     ['Jane Chernomorin'],
                     ['Noam Nativ'],
                     ['Mor Yesharim'],
                     ['Lucy Yanfital'],
                     ['Gal Derriere'],
                     ['Tal Doron'],
                     ['Ori Laslo'],
                     ['Omri Traub'],
                     ['Alon Fital'],
                     ['Ailon Velger'],
                     ['Boaz Arad'],
                     ['Roi Lipman']]
)

delete_friendships_query = QueryInfo(
    query="""MATCH (ME:person {name:'Roi Lipman'})-[e:friend]->() DELETE e""",
    description='Delete frienships',
    expected_result=[]
)

delete_person_query = QueryInfo(
    query="""MATCH (ME:person {name:'Roi Lipman'}) DELETE ME""",
    description='Delete myself from the graph',
    expected_result=[]
)

post_delete_label_query = QueryInfo(
    query="""MATCH (p:person) RETURN p.name""",
    description='Retrieve all nodes with person label',
    expected_result=[['Boaz Arad'],
                     ['Valerie Abigail Arad'],
                     ['Ori Laslo'],
                     ['Shelly Laslo Rooz'],
                     ['Ailon Velger'],
                     ['Noam Nativ'],
                     ['Jane Chernomorin'],
                     ['Alon Fital'],
                     ['Mor Yesharim'],
                     ['Gal Derriere'],
                     ['Lucy Yanfital'],
                     ['Tal Doron'],
                     ['Omri Traub']]
)

queries_info = [
    graph_entities,
    relation_type_counts,
    subset_of_people,
    my_friends_query,
    friends_of_friends_query,
    friends_of_friends_single_and_over_30_query,
    friends_of_friends_visited_netherlands_and_single_query,
    friends_visited_same_places_as_me_query,
    countries_visited_by_roi_tal_boaz,
    friends_older_than_me_query,
    friends_age_difference_query,
    friends_who_are_older_than_average,
    how_many_countries_each_friend_visited_query,
    visit_purpose_of_each_country_i_visited_query,
    who_was_on_business_trip_query,
    number_of_vacations_per_person_query,
    all_reachable_friends_query,
    all_reachable_countries_query,
    reachable_countries_or_people_query,
    all_reachable_countries_or_people_query,
    all_reachable_entities_query,
    all_reachable_people_min_2_hops_query,
    happy_birthday_query,
    friends_age_statistics_query,
    all_paths_leads_to_greece_query,
    number_of_paths_to_places_visited,
    pagerank_friends,
    delete_friendships_query,
    delete_person_query,
    post_delete_label_query
]
