################################
##  Query Expansion Prompts   ##
################################

depth_prompt = (
    """
    Given an initial base query, use your internal knowledge about Singapore to generate deeper and more specific queries related to the provided subject.

    Instructions:
    - Take the original query provided.
    - Generate {n} new queries that specifically deepen the subject matter. Expand “depth” by adding finer-grained facets (e.g. departments, programmes, policies, reports, initiatives, locations).
    - Focus exclusively on Singapore-specific context or examples.
    - Only when relevant and logical, incorporate any of the below topics found within the below labels in the query expansion. The provided labels are entity classes, infer what they mean and what they refer to: {bottom_labels}
    - Only provide the new queries, do not provide rationale or explanations 
    - Queries must **strictly** be about Singapore or must be related to Singapore 

    Example:
    Original Query: \"Home Team Science and Technology\"
    Generated Queries:
    1. \"Home Team Science and Technology departments Singapore\"
    2. \"Home Team Science and Technology research programmes Singapore\"
    3. \"Home Team Science and Technology innovation projects Singapore\"
    4. \"Home Team Science and Technology funding initiatives Singapore\"
    """
)

width_prompt = (
    """
    Given a root query, leverage your internal knowledge about Singapore to generate related but distinct queries within a complementary domain, explicitly relevant to Singapore.

    Instructions:
    - Take the original query provided.
    - Generate {n} complementary queries that are related but differ significantly from the original query. Explore a different but related topic (e.g. sibling dishes, parallel agencies, alternate programmes, related events).
    - Focus exclusively on Singapore-specific context or examples.
    - Only when relevant and logical, incorporate any of these subjects within the queries: {bottom_entities}
    - Only provide the new queries, do not provide rationale or explanations
    - Queries must **strictly** be about Singapore or must be related to Singapore
    
    Example:
    Original Query: \"Chicken rice\"
    Generated Complementary Queries:
    1. \"Chilli crab Singapore\"
    2. \"Hainanese curry rice Singapore\"
    3. \"Laksa stalls in Singapore\"
    4. \"Satay restaurants Singapore\"
    """
)
