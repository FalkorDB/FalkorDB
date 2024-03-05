import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from demo import QueryInfo


class IMDBQueries(object):
    def __init__(self, actors=None, movies=None):
        nodesAvailable = (movies is not None and actors is not None)
        
        ##################################################################
        ### number_of_actors_query
        ##################################################################

        self.number_of_actors_query = QueryInfo(
            query="""MATCH (n:actor) RETURN count(n) as actors_count""",
            description='How many actors are in the graph?',
            expected_result=[[1317]]
        )

        ##################################################################
        ### actors_played_with_nicolas_cage_query
        ##################################################################

        self.actors_played_with_nicolas_cage_query = QueryInfo(
            query="""MATCH (n:actor{name:"Nicolas Cage"})-[:act]->(m:movie)<-[:act]-(a:actor)
                    RETURN a.name, m.title ORDER BY a.name""",
            description='Which actors played along side Nicolas Cage?',
            expected_result=[['Cassi Thomson', 'Left Behind'],
                            ['Chad Michael Murray', 'Left Behind'],
                            ['Gary Grubbs', 'Left Behind'],
                            ['Jordin Sparks', 'Left Behind'],
                            ['Lea Thompson', 'Left Behind'],
                            ['Martin Klebba', 'Left Behind'],
                            ['Nicolas Cage', 'Left Behind'],
                            ['Quinton Aaron', 'Left Behind']]
        )


        ##################################################################
        ### find_three_actors_played_with_nicolas_cage_query
        ##################################################################

        self.find_three_actors_played_with_nicolas_cage_query = QueryInfo(
            query="""MATCH (nicolas:actor {name:"Nicolas Cage"})-[:act]->(m:movie)<-[:act]-(a:actor)
                    RETURN a.name, m.title
                    ORDER BY a.name, m.title
                    LIMIT 3""",
            description='Get 3 actors who have played along side Nicolas Cage?',
            expected_result=[['Cassi Thomson', 'Left Behind'],
                            ['Chad Michael Murray', 'Left Behind'],
                            ['Gary Grubbs', 'Left Behind']]
        )


        ##################################################################
        ### actors_played_in_movie_straight_outta_compton_query
        ##################################################################

        self.actors_played_in_movie_straight_outta_compton_query = QueryInfo(
            query="""MATCH (a:actor)-[:act]->(m:movie {title:"Straight Outta Compton"})
                    RETURN a.name""",
            description='Which actors played in the movie Straight Outta Compton?',
            expected_result=[['Aldis Hodge'],
                            ['Corey Hawkins'],
                            ['Neil Brown Jr.'],
                            ['O\'Shea Jackson Jr.']]
        )


        ##################################################################
        ### actors_over_50_that_played_in_blockbusters_query
        ##################################################################

        expected_result = None
        if nodesAvailable:
            expected_result=[
                [actors['Bill Irwin'], movies['Interstellar']],
                [actors['Vincent Price'], movies['Vincent']],
                [actors['Ellen Burstyn'], movies['Interstellar']],
                [actors['Paul Reiser'], movies['Whiplash']],
                [actors['Francis X. McCarthy'], movies['Interstellar']],
                [actors['John Lithgow'], movies['Interstellar']],
                [actors['J.K. Simmons'], movies['Whiplash']],
                [actors['Chris Mulkey'], movies['Whiplash']],
                [actors['Rachael Harris'], movies['Lucifer']],
                [actors['Matthew McConaughey'], movies['Interstellar']],
                [actors['D.B. Woodside'], movies['Lucifer']]
            ]

        self.actors_over_50_that_played_in_blockbusters_query = QueryInfo(
            query="""MATCH (a:actor)-[:act]->(m:movie)
                    WHERE a.age >= 50 AND m.votes > 10000 AND m.rating > 8.2
                    RETURN a, m ORDER BY a.name, m.name""",
            description='Which actors who are over 50 played in blockbuster movies?',
            expected_result=expected_result
        )

        ##################################################################
        ### actors_played_in_bad_drama_or_comedy_query
        ##################################################################
        
        expected_result = None
        if nodesAvailable:
            expected_result=[
                ['Rita Ora', movies['Fifty Shades of Grey']],
                ['Dakota Johnson', movies['Fifty Shades of Grey']],
                ['Marcia Gay Harden', movies['Fifty Shades of Grey']],
                ['Jamie Dornan', movies['Fifty Shades of Grey']],
                ['Eloise Mumford', movies['Fifty Shades of Grey']],
                ['Max Martini', movies['Fifty Shades of Grey']],
                ['Luke Grimes', movies['Fifty Shades of Grey']],
                ['Jennifer Ehle', movies['Fifty Shades of Grey']],
                ['Victor Rasuk', movies['Fifty Shades of Grey']],
                ['Nancy Lenehan', movies['Sex Tape']],
                ['Rob Lowe', movies['Sex Tape']],
                ['Cameron Diaz', movies['Sex Tape']],
                ['Rob Corddry', movies['Sex Tape']],
                ['Jason Segel', movies['Sex Tape']],
                ['Ellie Kemper', movies['Sex Tape']]
            ]

        self.actors_played_in_bad_drama_or_comedy_query = QueryInfo(
            query="""MATCH (a:actor)-[:act]->(m:movie)
                    WHERE (m.genre = "Drama" OR m.genre = "Comedy")
                    AND m.rating < 5.5 AND m.votes > 50000
                    RETURN a.name, m
                    ORDER BY m.rating""",
            description='Which actors played in bad drama or comedy?',
            expected_result = expected_result            
        )

        ##################################################################
        ### actors_played_in_drama_action_comedy
        ##################################################################
        
        expected_result=[
                ['Bradley Cooper'],
                ['Michael B. Jordan'],
                ['Michael Caine'],
                ['Miles Teller']
            ]

        self.actors_played_in_drama_action_comedy_query = QueryInfo(
            query="""MATCH (a:actor)-[:act]->(m0:movie {genre:'Action'}),
                           (a)-[:act]->(m1:movie {genre:'Drama'}),
                           (a)-[:act]->(m2:movie {genre:'Comedy'})
                    RETURN DISTINCT a.name""",
            description='Which actors played in Action, Drama and Comedy movies?',
            reversible=False,
            expected_result = expected_result            
        )

        ##################################################################
        ### young_actors_played_with_cameron_diaz_query
        ##################################################################
        
        expected_result = None
        if nodesAvailable:
            expected_result=[
                [actors['Nicolette Pierini'], 'Annie'],
                [actors['Kate Upton'], 'The Other Woman']
            ]

        self.young_actors_played_with_cameron_diaz_query = QueryInfo(
            query="""MATCH (Cameron:actor {name:"Cameron Diaz"})-[:act]->(m:movie)<-[:act]-(a:actor)
                    WHERE a.age < 35
                    RETURN a, m.title ORDER BY m.title""",
            description='Which young actors played along side Cameron Diaz?',
            expected_result=expected_result
        )

        
        ##################################################################
        ### actors_played_with_cameron_diaz_and_younger_than_her_query
        ##################################################################

        expected_result = None
        if nodesAvailable:
            expected_result=[
                [actors['Jason Segel'], 'Sex Tape'],
                [actors['Ellie Kemper'], 'Sex Tape'],
                [actors['Nicolette Pierini'], 'Annie'],
                [actors['Rose Byrne'], 'Annie'],
                [actors['Kate Upton'], 'The Other Woman'],
                [actors['Nicki Minaj'], 'The Other Woman'],
                [actors['Taylor Kinney'], 'The Other Woman']
            ]

        self.actors_played_with_cameron_diaz_and_younger_than_her_query = QueryInfo(
            query="""MATCH (Cameron:actor {name:"Cameron Diaz"})-[:act]->(m:movie)<-[:act]-(a:actor)
                    WHERE a.age < Cameron.age
                    RETURN a, m.title order by a.name""",
            description='Which actors played along side Cameron Diaz and are younger then her?',
            expected_result=expected_result
        )


        ##################################################################
        ### sum_and_average_age_of_straight_outta_compton_cast_query
        ##################################################################

        self.sum_and_average_age_of_straight_outta_compton_cast_query = QueryInfo(
            query="""MATCH (a:actor)-[:act]->(m:movie{title:"Straight Outta Compton"})
                    RETURN m.title, SUM(a.age), AVG(a.age)""",
            description='What is the sum and average age of the Straight Outta Compton cast?',
            expected_result=[['Straight Outta Compton', 131, 32.75]]
        )


        ##################################################################
        ### how_many_movies_cameron_diaz_played_query
        ##################################################################

        self.how_many_movies_cameron_diaz_played_query = QueryInfo(
            query="""MATCH (Cameron:actor{name:"Cameron Diaz"})-[:act]->(m:movie)
                    RETURN Cameron.name, COUNT(m.title)""",
            description='In how many movies did Cameron Diaz played?',
            expected_result=[['Cameron Diaz', 3]]
        )

        
        ##################################################################
        ### find_ten_oldest_actors_query
        ##################################################################

        expected_result = None
        
        expected_result = None
        if nodesAvailable:
            expected_result=[
                [actors['Vincent Price']],
                [actors['George Kennedy']],
                [actors['Cloris Leachman']],
                [actors['John Cullum']],
                [actors['Lois Smith']],
                [actors['Robert Duvall']],
                [actors['Olympia Dukakis']],
                [actors['Ellen Burstyn']],
                [actors['Michael Caine']],
                [actors['Judi Dench']]
            ]

        self.find_ten_oldest_actors_query = QueryInfo(
            query="""MATCH (a:actor)
                    RETURN DISTINCT *
                    ORDER BY a.age DESC
                    LIMIT 10""",
            description='10 Oldest actors?',
            expected_result=expected_result            
        )

        ##################################################################
        ### actors_over_85_index_scan
        ##################################################################

        expected_result = None
        if nodesAvailable:
            expected_result=[
                [actors['Michael Caine']],
                [actors['Ellen Burstyn']],
                [actors['Robert Duvall']],
                [actors['Olympia Dukakis']],
                [actors['Lois Smith']],
                [actors['John Cullum']],
                [actors['Cloris Leachman']],
                [actors['George Kennedy']],
                [actors['Vincent Price']]
            ]

        self.actors_over_85_index_scan = QueryInfo(
            query="""MATCH (a:actor)
                    WHERE a.age > 85
                    RETURN *
                    ORDER BY a.age, a.name""",
            description='Actors over 85 on indexed property?',
            expected_result=expected_result
        )

        ##################################################################
        ### eighties_movies_index_scan
        ##################################################################

        self.eighties_movies_index_scan = QueryInfo(
            query="""MATCH (m:movie)
                    WHERE 1980 <= m.year
                    AND m.year < 1990
                    RETURN m.title, m.year
                    ORDER BY m.year""",
            description='Multiple filters on indexed property?',
            expected_result=[['The Evil Dead', 1981],
                            ['Vincent', 1982]]
        )

        ##################################################################
        ### find_titles_starting_with_american_query
        ##################################################################

        self.find_titles_starting_with_american_query = QueryInfo(
            query="""MATCH (m:movie)
                    WHERE LEFT(m.title, 8) = 'American'
                    RETURN m.title
                    ORDER BY m.title""",
            description='Movies starting with "American"?',
            expected_result=[['American Honey'],
                            ['American Pastoral'],
                            ['American Sniper']]
        )

        ##################################################################
        ### same_year_higher_rating_than_huntforthewilderpeople
        ##################################################################

        self.same_year_higher_rating_than_huntforthewilderpeople_query = QueryInfo(
            query="""MATCH (base:movie), (option:movie)
                     WHERE base.title = 'Hunt for the Wilderpeople' AND
                     base.year = option.year AND
                     option.rating > base.rating
                     RETURN option.title, option.rating
                     ORDER BY option.rating, option.title desc
                     LIMIT 10""",
            description='List 10 movies released on the same year as "Hunt for the Wilderpeople" that got higher rating than it',
            reversible=False,
            expected_result=[["Hacksaw Ridge", 8.8],
                             ["Moonlight", 8.7],
                             ["La La Land", 8.6],
                             ["Arrival", 8.5],
                             ["Hell or High Water", 8.2],
                             ["Zootopia", 8.1],
                             ["Split", 8.1],
                             ["Nocturnal Animals", 8.1],
                             ["Deadpool", 8.1],
                             ["Ah-ga-ssi", 8.1]]
        )

        ##################################################################
        ### all_actors_named_tim
        ##################################################################

        self.all_actors_named_tim = QueryInfo(
            query="""CALL db.idx.fulltext.queryNodes('actor', 'tim') YIELD node
                     RETURN node.name
                     ORDER BY node.name""",
            description='All actors named Tim',
            reversible=False,
            expected_result=[['Tim Roth'],
                            ['Tim Reid'],
                            ['Tim McGraw'],
                            ['Tim Griffin'],
                            ['Tim Blake Nelson']]
        )

        ##################################################################
        ### grand_budapest_hotel_cast_and_their_other_roles
        ##################################################################

        self.grand_budapest_hotel_cast_and_their_other_roles = QueryInfo(

            query="""MATCH (a:actor)-[:act]->(h:movie {title: 'The Grand Budapest Hotel'})
                     OPTIONAL MATCH (a)-[:act]->(m:movie) WHERE m <> h
                     RETURN a.name, m.title
                     ORDER BY a.name, m.title""",

            description='All actors in The Grand Budapest Hotel and their other movies',
            reversible=False,
            expected_result=[['Adrien Brody', None],
                             ['Bill Murray', 'The Jungle Book'],
                             ['F. Murray Abraham', None],
                             ['Harvey Keitel', 'The Ridiculous 6'],
                             ['Harvey Keitel', 'Youth'],
                             ['Jeff Goldblum', 'Independence Day: Resurgence'],
                             ['Jude Law', 'Spy'],
                             ['Mathieu Amalric', None],
                             ['Ralph Fiennes', 'A Bigger Splash'],
                             ['Ralph Fiennes', 'Spectre'],
                             ['Willem Dafoe', 'John Wick'],
                             ['Willem Dafoe', 'The Fault in Our Stars']]
        )

        self.queries_info = [
            self.number_of_actors_query,
            self.actors_played_with_nicolas_cage_query,
            self.find_three_actors_played_with_nicolas_cage_query,
            self.actors_played_in_movie_straight_outta_compton_query,
            self.actors_over_50_that_played_in_blockbusters_query,
            self.actors_played_in_bad_drama_or_comedy_query,
            self.actors_played_in_drama_action_comedy_query,
            self.young_actors_played_with_cameron_diaz_query,
            self.actors_played_with_cameron_diaz_and_younger_than_her_query,
            self.sum_and_average_age_of_straight_outta_compton_cast_query,
            self.how_many_movies_cameron_diaz_played_query,
            self.find_ten_oldest_actors_query,
            self.actors_over_85_index_scan,
            self.eighties_movies_index_scan,
            self.find_titles_starting_with_american_query,
            self.same_year_higher_rating_than_huntforthewilderpeople_query,
            self.all_actors_named_tim,
            self.grand_budapest_hotel_cast_and_their_other_roles
        ]

    def queries(self):
        return self.queries_info
