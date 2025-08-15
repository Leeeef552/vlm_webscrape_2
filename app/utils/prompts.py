################################
##  Query Expansion Prompts   ##
################################

entity_prompt = (
"""
    ## Context:
    You are an expert in Singapore knowledge generating Search Engine queries to expand a Singapore-focused knowledge base. Starting from an Original Query, produce two sets of Singapore-specific queries: (A) Expanded queries that add finer-grained facets about the same topic, and (B) Complementary queries that explore related but distinct entities/topics within the same subdomain in Singapore.

    ## Instructions:
    - Use the Original Query provided.
    - Generate Expanded Queries that add finer-grained facets (e.g., departments, programmes, policies, reports, initiatives, locations) and Complementary Queries that are related but differ significantly (e.g., sibling dishes, parallel agencies, alternate programmes, related events).
    - Generate {n} queries in total
    - Focus exclusively on Singapore-specific context or examples.
    - Only provide the new queries; do not provide rationale or explanations.
    - Queries must **strictly** be about or related to Singapore.

    ## Example:
    Original Query: "Home Team Science and Technology"
    Expanded Queries:
    1. "Home Team Science and Technology departments Singapore"
    2. "Home Team Science and Technology research programmes Singapore"
    3. "Home Team Science and Technology innovation projects Singapore"
    4. "Home Team Science and Technology funding initiatives Singapore"
    Complementary Queries:
    1. "Defence Science and Technology Agency programmes Singapore"
    2. "GovTech innovation labs Singapore"
    3. "Health Sciences Authority forensic science services Singapore"
    4. "National Research Foundation RIE2025 security and resilience Singapore"
    """
)

labels_prompt = (
    """
    ## Context:
    You are an expert at Singapore knowledge and you are creating (or any other) Search Engine queries to help source for information to expand a Singapore-focused knowledge base. 

    ## Instructions:
    - Generate {n} Singapore-related queries that explore specific labels or categories associated with the original query.
    - These labels should reflect thematic groupings, roles, types, or classifications relevant to the seed query.
    - Focus exclusively on Singapore-specific context or examples.
    - Only provide the new queries, do not provide rationale or explanations
    - Queries must **strictly** be about Singapore or must be related to Singapore
    
    ## Example:
    Original Query: "Government Agencies"
    Generated Label Queries:
    1. "Central Provident Fund"
    2. "Home Team Science and Technology"
    3. "Health Science Authority"
    4. "Urban Redevelopment Authority"
    """
)

