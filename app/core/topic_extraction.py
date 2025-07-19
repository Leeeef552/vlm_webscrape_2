from llama_index.core import Document, PropertyGraphIndex, StorageContext
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.prompts import PromptTemplate
import json
import re
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import networkx as nx
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from pathlib import Path
from llama_index.core import Document, PropertyGraphIndex, StorageContext
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
import re
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import networkx as nx
import os

# Initialize LLM settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # small & fast
)

os.environ["OPENAI_API_KEY"] = "dummy"  # required even if not used
os.environ["OPENAI_BASE_API"] = "http://localhost:8124/v1"

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = OpenAILike(
    model="unsloth/Llama-3.2-3B-Instruct",
    api_base="http://localhost:8124/v1",
    is_chat_model=False,  # set True if only chat endpoint exists
)

class AutonomousSingaporeKG:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.index = None
        self.query_engine = None
        
        # Track exploration state
        self.discovered_entities = set()
        self.explored_search_queries = set()
        self.entity_relationships = defaultdict(list)
        self.knowledge_gaps = []
        self.expansion_frontier = []
        
        # Enhanced extraction focusing on entities and relationships
        self.triplet_prompt = PromptTemplate(
            """You are analyzing Singapore content to build a comprehensive knowledge graph.
            Extract **up to {max_paths_per_chunk}** knowledge triples focusing on:
            
            PRIORITY ENTITIES (extract these first):
            • Government agencies, ministries, statutory boards
            • Companies, organizations, institutions
            • Places, districts, landmarks, infrastructure
            • Policies, programs, initiatives, schemes
            • People, leaders, key figures
            • Economic sectors, industries
            • Cultural elements, events, traditions
            
            Format: (Subject, Predicate, Object) — one per line
            
            Guidelines:
            • Use specific Singapore entity names (not generic terms)
            • Include cross-domain relationships (e.g., policy → economic impact)
            • Capture temporal relationships (before, after, during)
            • Link abstract concepts to concrete entities
            
            Text:
            ----
            {context_str}
            ----
            
            Triples:"""
        )
        
        # Entity gap analysis prompt
        self.gap_analysis_prompt = PromptTemplate(
            """Analyze the following Singapore knowledge graph entities and relationships to identify KNOWLEDGE GAPS and UNEXPLORED CONNECTIONS.
            
            Current entities: {entities}
            Current relationships: {relationships}
            
            Identify:
            1. **Missing Cross-Sector Connections**: What relationships between different domains (govt-business, education-tech, etc.) are missing?
            2. **Underexplored Entities**: Which important Singapore entities are mentioned but lack detailed coverage?
            3. **Temporal Gaps**: What historical developments or recent changes need exploration?
            4. **Geographic Gaps**: Which regions, districts, or locations need more coverage?
            5. **Stakeholder Gaps**: Which organizations, companies, or institutions are referenced but underexplored?
            
            Generate 5-8 specific search queries that would fill these gaps:
            Format as JSON: {{"search_queries": ["query1", "query2", ...], "rationale": ["reason1", "reason2", ...]}}
            """
        )
        
        # Entity expansion prompt
        self.entity_expansion_prompt = PromptTemplate(
            """Given this Singapore entity: "{entity}"
            And its current context: {context}
            
            Generate search queries to discover:
            1. **Deeper Details**: Internal structure, components, sub-entities
            2. **Broader Context**: Parent organizations, wider ecosystems
            3. **Relationships**: Partners, competitors, collaborators
            4. **Impact**: Beneficiaries, stakeholders, effects
            5. **Evolution**: History, changes, future plans
            6. **Cross-References**: Similar entities in other sectors
            
            Output 4-6 specific search queries as JSON list:
            {{"queries": ["query1", "query2", ...]}}
            """
        )
    
    def load_and_build_kg(self):
        """Load documents and build knowledge graph with entity tracking"""
        print("📁 Loading documents...")
        docs = self._load_documents()
        
        if not docs:
            print("❌ No documents loaded. Please check your data directory.")
            return False
            
        print(f"✅ Loaded {len(docs)} documents")
        
        # Show sample of document content for debugging
        if docs:
            sample_doc = docs[0]
            print(f"📄 Sample document preview:")
            print(f"   Title: {sample_doc.metadata.get('title', 'N/A')}")
            print(f"   Text length: {len(sample_doc.text)}")
            print(f"   Text preview: {sample_doc.text[:200]}...")
        
        print("🏗️ Building knowledge graph...")
        
        # Create extractor with entity-focused prompt
        extractor = SimpleLLMPathExtractor(
            llm=Settings.llm,
            max_paths_per_chunk=60,  # Increased for more entity extraction
            extract_prompt=self.triplet_prompt,
        )
        
        try:
            # Build the property graph index
            self.index = PropertyGraphIndex.from_documents(
                docs,
                kg_extractors=[extractor],
            )
            
            # Create query engine optimized for entity discovery
            self.query_engine = self.index.as_query_engine(
                include_text=True,
                response_mode="compact",
                similarity_top_k=15
            )
            
            print("✅ Knowledge graph built successfully!")
            
            # Extract and analyze entities from the built graph
            self._extract_entities_from_graph()
            
            if len(self.discovered_entities) == 0:
                print("⚠️  No entities discovered. This might indicate:")
                print("   - Document text content is empty or too short")
                print("   - Entity extraction patterns need adjustment")
                print("   - LLM connection issues during graph building")
                
                # Try direct text analysis as fallback
                print("🔄 Attempting direct text analysis...")
                for doc in docs[:3]:  # Analyze first 3 docs
                    entities = self._extract_entities_from_text(doc.text)
                    self.discovered_entities.update(entities)
                    print(f"   Found {len(entities)} entities in document")
            
            print(f"📊 Final count: {len(self.discovered_entities)} entities discovered")
            if self.discovered_entities:
                print(f"🎯 Sample entities: {list(self.discovered_entities)[:10]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error building knowledge graph: {e}")
            return False
    
    def _load_documents(self) -> List[Document]:
        """Load documents with enhanced metadata for entity tracking"""
        docs = []
        
        print(f"📂 Looking for documents in: {self.data_dir}")
        
        if self.data_dir.is_file():
            json_files = [self.data_dir]
            print(f"📄 Processing single file: {self.data_dir}")
        else:
            json_files = list(self.data_dir.glob("*.json"))
            print(f"📄 Found {len(json_files)} JSON files")
        
        total_processed = 0
        total_skipped = 0
        
        for fp in json_files:
            try:
                print(f"   Processing: {fp.name}")
                with fp.open(encoding="utf-8") as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    data = [data]
                
                file_processed = 0
                file_skipped = 0
                
                for obj in data:
                    txt = obj.get("text_content", "")
                    
                    # More lenient content filtering
                    if not txt or len(txt.strip()) < 50:  # Reduced from 100
                        file_skipped += 1
                        continue
                    
                    # Show sample text for debugging
                    if file_processed == 0 and len(docs) < 3:
                        print(f"      Sample text: {txt[:150]}...")
                    
                    docs.append(Document(
                        text=self._clean_text(txt),
                        metadata={
                            "file": fp.name,
                            "url": obj.get("url", ""),
                            "title": obj.get("title", ""),
                            "timestamp": obj.get("timestamp", ""),
                            "domain": self._extract_domain_from_url(obj.get("url", ""))
                        }
                    ))
                    file_processed += 1
                
                total_processed += file_processed
                total_skipped += file_skipped
                print(f"      ✅ Processed: {file_processed}, Skipped: {file_skipped}")
                
            except Exception as e:
                print(f"      ❌ Error loading {fp}: {e}")
        
        print(f"📊 Total documents processed: {total_processed}, skipped: {total_skipped}")
        return docs
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving entity names"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,;:!?&()-]', ' ', text)
        return text.strip()
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain category from URL for context"""
        if not url:
            return "unknown"
        
        domain_patterns = {
            'government': ['gov.sg', 'moe.gov.sg', 'mti.gov.sg', 'mnd.gov.sg'],
            'news': ['channelnewsasia', 'straitstimes', 'todayonline'],
            'business': ['businesstimes', 'bloomberg', 'reuters'],
            'academic': ['edu.sg', 'nus.edu', 'ntu.edu'],
            'statutory': ['hdb.gov.sg', 'cpf.gov.sg', 'iras.gov.sg']
        }
        
        for category, patterns in domain_patterns.items():
            if any(pattern in url.lower() for pattern in patterns):
                return category
        return "general"
    
    def _extract_entities_from_graph(self):
        """Extract entities from the built knowledge graph"""
        print("🔍 Extracting entities from knowledge graph...")
        
        # First, try to extract entities directly from document text
        self._extract_entities_from_documents()
        
        # Then try querying the graph (if it works)
        if self.query_engine:
            entity_queries = [
                "What are the key organizations mentioned?",
                "What locations are discussed?",
                "What government entities are referenced?",
                "What companies or businesses are mentioned?",
                "What policies or programs are described?"
            ]
            
            for query in entity_queries:
                try:
                    print(f"  Querying: {query[:50]}...")
                    response = self.query_engine.query(query)
                    response_text = str(response)
                    print(f"  Response length: {len(response_text)}")
                    
                    # Extract entities from response
                    entities = self._extract_entities_from_text(response_text)
                    print(f"  Found {len(entities)} entities")
                    self.discovered_entities.update(entities)
                    
                except Exception as e:
                    print(f"  Error in query '{query[:30]}...': {e}")
        
        print(f"✅ Total entities discovered: {len(self.discovered_entities)}")
        if self.discovered_entities:
            print(f"Sample entities: {list(self.discovered_entities)[:10]}")
    
    def _extract_entities_from_documents(self):
        """Extract entities directly from loaded documents"""
        print("📄 Extracting entities from raw documents...")
        
        # Re-read documents to extract entities
        docs = self._load_documents()
        entity_count = 0
        
        for doc in docs:
            entities = self._extract_entities_from_text(doc.text)
            self.discovered_entities.update(entities)
            entity_count += len(entities)
            
        print(f"📊 Extracted {entity_count} entity mentions from {len(docs)} documents")
    
    def _extract_entities_from_text(self, text: str) -> Set[str]:
        """Extract potential entities from text using comprehensive patterns"""
        entities = set()
        
        # More comprehensive Singapore-specific patterns
        patterns = [
            # Government entities
            r'\b(?:Ministry of|MOE|MAS|HDB|CPF|IRAS|URA|BCA|NEA|PUB|LTA|SLA|MND|MTI|MHA|MINDEF|MOM|MSF|MCCY)\b',
            r'\b[A-Z][a-zA-Z&\s]{3,40}(?:Ministry|Agency|Authority|Board|Council|Commission)\b',
            
            # Companies and organizations
            r'\b[A-Z][a-zA-Z&\s]{2,30}(?:Pte Ltd|Ltd|Corporation|Corp|Holdings|Group|Company)\b',
            r'\b(?:DBS|OCBC|UOB|SingTel|StarHub|M1|CapitaLand|City Developments|Wilmar|Olam)\b',
            
            # Educational institutions
            r'\b(?:NUS|NTU|SMU|SUSS|SIT|SUTD|Nanyang|National University)\b',
            r'\b[A-Z][a-zA-Z\s&]{3,25}(?:University|Institute|School|College|Academy|Polytechnic)\b',
            
            # Places and locations
            r'\b[A-Z][a-zA-Z\s]{2,25}(?:District|Road|Avenue|Street|Drive|Park|Centre|Center|Plaza|Mall|Estate|Town|Hub)\b',
            r'\b(?:Orchard|Marina Bay|Sentosa|Jurong|Tampines|Woodlands|Ang Mo Kio|Toa Payoh|Bedok|Hougang)\b',
            
            # Programs and initiatives
            r'\b(?:Singapore|SG)\s+[A-Za-z\s]{3,30}(?:Initiative|Program|Programme|Scheme|Plan|Policy|Strategy)\b',
            r'\b[A-Z][a-zA-Z\s]{3,30}(?:Initiative|Program|Programme|Scheme|Plan|Policy|Strategy)\b',
            
            # Technology and innovation
            r'\b(?:Smart Nation|Digital|FinTech|GovTech|IMDA|A\*STAR|CREATE|SUTD)\b',
            
            # General proper nouns (capitalized phrases)
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',
            
            # Acronyms and abbreviations
            r'\b[A-Z]{2,6}\b'
        ]
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    match = match.strip()
                    # Filter out common words and very short matches
                    if (len(match) > 2 and 
                        not match.lower() in {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'they', 'have', 'been', 'were', 'said', 'more', 'also', 'such', 'may', 'can', 'will', 'new', 'all'} and
                        not match.isdigit()):
                        entities.add(match)
            except Exception as e:
                print(f"Error with pattern {pattern}: {e}")
        
        return entities
    
    def discover_knowledge_gaps(self) -> List[str]:
        """Analyze current knowledge to find gaps and generate search queries"""
        print("🔍 Analyzing knowledge gaps...")
        
        # Prepare current state summary
        entities_sample = list(self.discovered_entities)[:50]  # Sample for context
        relationships_sample = list(self.entity_relationships.keys())[:20]
        
        try:
            prompt = self.gap_analysis_prompt.format(
                entities=", ".join(entities_sample),
                relationships=", ".join(relationships_sample)
            )
            
            response = Settings.llm.complete(prompt)
            response_text = str(response)
            
            # Extract search queries from JSON response
            queries = self._extract_json_queries(response_text)
            
            if not queries:
                # Fallback: generate queries based on entity analysis
                queries = self._generate_fallback_gap_queries()
            
            # Filter out already explored queries
            new_queries = [q for q in queries if q not in self.explored_search_queries]
            self.explored_search_queries.update(new_queries)
            
            print(f"Generated {len(new_queries)} gap-filling queries")
            return new_queries
            
        except Exception as e:
            print(f"Error in gap analysis: {e}")
            return self._generate_fallback_gap_queries()
    
    def expand_entity_knowledge(self, entity: str, max_queries: int = 5) -> List[str]:
        """Generate search queries to expand knowledge about a specific entity"""
        print(f"🎯 Expanding knowledge for entity: {entity}")
        
        # Get current context for this entity
        try:
            context_query = f"What do we know about {entity} in Singapore?"
            context_response = self.query_engine.query(context_query)
            context = str(context_response)[:800]  # Limit context size
        except:
            context = f"Limited information available about {entity}"
        
        try:
            prompt = self.entity_expansion_prompt.format(
                entity=entity,
                context=context
            )
            
            response = Settings.llm.complete(prompt)
            queries = self._extract_json_queries(str(response))
            
            if not queries:
                # Generate basic expansion queries
                queries = [
                    f"{entity} Singapore structure organization",
                    f"{entity} Singapore partnerships collaborations",
                    f"{entity} Singapore impact benefits",
                    f"{entity} Singapore history development",
                    f"{entity} Singapore future plans initiatives"
                ]
            
            # Filter and limit
            new_queries = [q for q in queries[:max_queries] if q not in self.explored_search_queries]
            self.explored_search_queries.update(new_queries)
            
            print(f"Generated {len(new_queries)} expansion queries for {entity}")
            return new_queries
            
        except Exception as e:
            print(f"Error expanding entity {entity}: {e}")
            return []
    
    def get_autonomous_search_queries(self, num_queries: int = 10) -> Dict[str, List[str]]:
        """Main method to get search queries for autonomous exploration"""
        all_queries = {
            "gap_queries": [],
            "entity_expansion_queries": [],
            "cross_domain_queries": []
        }
        
        # 1. Discover knowledge gaps (40% of queries)
        gap_queries = self.discover_knowledge_gaps()
        all_queries["gap_queries"] = gap_queries[:int(num_queries * 0.4)]
        
        # 2. Expand top entities (40% of queries)
        top_entities = list(self.discovered_entities)[:5]  # Focus on top entities
        entity_queries = []
        for entity in top_entities:
            entity_queries.extend(self.expand_entity_knowledge(entity, max_queries=2))
        all_queries["entity_expansion_queries"] = entity_queries[:int(num_queries * 0.4)]
        
        # 3. Generate cross-domain exploration queries (20% of queries)
        cross_domain = self._generate_cross_domain_queries()
        all_queries["cross_domain_queries"] = cross_domain[:int(num_queries * 0.2)]
        
        # Flatten and return all queries
        flat_queries = []
        for category, queries in all_queries.items():
            flat_queries.extend(queries)
        
        print(f"\n🚀 Generated {len(flat_queries)} autonomous search queries:")
        print(f"  - Gap analysis: {len(all_queries['gap_queries'])}")
        print(f"  - Entity expansion: {len(all_queries['entity_expansion_queries'])}")
        print(f"  - Cross-domain: {len(all_queries['cross_domain_queries'])}")
        
        return {
            "queries": flat_queries[:num_queries],
            "breakdown": all_queries
        }
    
    def _extract_json_queries(self, text: str) -> List[str]:
        """Extract queries from JSON response"""
        try:
            # Try to find JSON structure
            import json
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'queries' in data:
                    return data['queries']
                elif 'search_queries' in data:
                    return data['search_queries']
            
            # Fallback: extract quoted strings
            queries = re.findall(r'"([^"]+)"', text)
            return [q for q in queries if len(q) > 10 and 'singapore' in q.lower()]
            
        except:
            return []
    
    def _generate_fallback_gap_queries(self) -> List[str]:
        """Generate basic gap-filling queries when AI generation fails"""
        return [
            "Singapore government digital transformation initiatives",
            "Singapore startup ecosystem funding support",
            "Singapore smart city infrastructure projects",
            "Singapore education technology integration",
            "Singapore healthcare system innovations",
            "Singapore sustainability environmental policies",
            "Singapore international trade partnerships"
        ]
    
    def _generate_cross_domain_queries(self) -> List[str]:
        """Generate queries exploring cross-domain connections"""
        domains = ['government', 'business', 'education', 'technology', 'healthcare', 'environment']
        queries = []
        
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                queries.append(f"Singapore {domain1} {domain2} collaboration partnership")
        
        return queries[:5]
    
    def update_knowledge_graph(self, new_documents: List[Document]):
        """Update the knowledge graph with new documents"""
        print(f"📈 Updating knowledge graph with {len(new_documents)} new documents...")
        
        # This would require rebuilding or incrementally updating the index
        # For now, we'll rebuild - in production, consider incremental updates
        all_docs = self._load_documents() + new_documents
        
        extractor = SimpleLLMPathExtractor(
            llm=Settings.llm,
            max_paths_per_chunk=60,
            extract_prompt=self.triplet_prompt,
        )
        
        self.index = PropertyGraphIndex.from_documents(
            all_docs,
            kg_extractors=[extractor],
        )
        
        self.query_engine = self.index.as_query_engine(
            include_text=True,
            response_mode="compact",
            similarity_top_k=15
        )
        
        # Re-extract entities
        self._extract_entities_from_graph()
        
        print(f"✅ Knowledge graph updated! Now tracking {len(self.discovered_entities)} entities")
    
    def get_exploration_status(self) -> Dict[str, Any]:
        """Get current exploration status and statistics"""
        return {
            "discovered_entities_count": len(self.discovered_entities),
            "explored_queries_count": len(self.explored_search_queries),
            "sample_entities": list(self.discovered_entities)[:20],
            "recent_queries": list(self.explored_search_queries)[-10:],
            "knowledge_domains": self._analyze_knowledge_domains()
        }
    
    def _analyze_knowledge_domains(self) -> Dict[str, int]:
        """Analyze what domains are covered in current knowledge"""
        domain_keywords = {
            'government': ['ministry', 'agency', 'policy', 'government'],
            'business': ['company', 'business', 'industry', 'economic'],
            'education': ['school', 'university', 'education', 'learning'],
            'technology': ['digital', 'tech', 'innovation', 'smart'],
            'infrastructure': ['transport', 'housing', 'infrastructure', 'urban'],
            'healthcare': ['health', 'medical', 'hospital', 'care']
        }
        
        domain_counts = defaultdict(int)
        
        for entity in self.discovered_entities:
            entity_lower = entity.lower()
            for domain, keywords in domain_keywords.items():
                if any(keyword in entity_lower for keyword in keywords):
                    domain_counts[domain] += 1
        
        return dict(domain_counts)

# Usage for autonomous exploration
def main():
    DATA_DIR = Path("/home/leeeefun681/volume/eefun/webscraping/scraping/vlm_webscrape/app/storage/text_data/text_markdown.json")
    
    # Initialize autonomous KG system
    kg_system = AutonomousSingaporeKG(DATA_DIR)
    
    # Build initial knowledge graph
    print("🏗️ Building autonomous Singapore knowledge graph...")
    kg_system.load_and_build_kg()
    
    # Get exploration status
    status = kg_system.get_exploration_status()
    print(f"\n📊 Current Knowledge State:")
    print(f"  - Entities discovered: {status['discovered_entities_count']}")
    print(f"  - Domain coverage: {status['knowledge_domains']}")
    
    # Generate autonomous search queries
    print("\n🤖 Generating autonomous search queries...")
    search_results = kg_system.get_autonomous_search_queries(num_queries=12)
    
    print(f"\n📋 Search Queries for SearXNG:")
    print("=" * 50)
    for i, query in enumerate(search_results["queries"], 1):
        print(f"{i:2d}. {query}")
    
    # Show breakdown
    print(f"\n🔍 Query Breakdown:")
    for category, queries in search_results["breakdown"].items():
        if queries:
            print(f"\n{category.replace('_', ' ').title()}:")
            for q in queries:
                print(f"  • {q}")
    
    return search_results["queries"]

if __name__ == "__main__":
    queries_for_searxng = main()