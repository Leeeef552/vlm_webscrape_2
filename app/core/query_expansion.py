from ..utils.logger import logger
from ..configs.config import QueryExpansionConfig
from pathlib import Path
import json
import re
from openai import OpenAI
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


class QueryExpansion:
    def __init__(self, config: QueryExpansionConfig):
        self.config = config
        self.data_path = Path(self.config.data_path)

        # Initialize OpenAI client once
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key="dummy"
        )

        # Prompt templates
        self.depth_prompt = (
            """
            Given an initial base query, use your internal knowledge about Singapore to generate deeper and more specific queries related to the provided subject.

            Instructions:
            - Take the original query provided.
            - Generate {n} new queries that specifically deepen the subject matter.
            - Focus exclusively on Singapore-specific context or examples.
            - Only when relevant and logical, incorporate any of the below topics found within the below labels in the query expansion. The provided labels are entitiy classes, infer what they mean and what they refer to: {bottom_labels}
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

        self.width_prompt = (
            """
            Given a root query, leverage your internal knowledge about Singapore to generate related but distinct queries within a complementary domain, explicitly relevant to Singapore.

            Instructions:
            - Take the original query provided.
            - Generate {n} complementary queries that are related but differ significantly from the original query.
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

    def _parse_queries(self, text: str) -> List[str]:
        """
        Parse numbered or quoted lines from the LLM response into a list of query strings.
        """
        queries: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            # Match lines like '1. "query"' or '1) query'
            m = re.match(r'^\d+[\)\.\-\s]+["“]?(.*?)["”]?$', line)
            if m:
                queries.append(m.group(1).strip())
        return queries

    def get_bottom_labels(self, n: int = 5) -> List[Tuple[str, int]]:
        """
        Return the bottom-n labels (categories) by total count.
        """
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        label_counts = {label: sum(entities.values()) for label, entities in data.items()}
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1])
        return sorted_labels[:n]

    def get_bottom_entities(self, n: int = 5) -> List[Tuple[str, int]]:
        """
        Return the bottom-n individual entities by count.
        """
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        entity_counts = {
            entity: count
            for entities in data.values()
            for entity, count in entities.items()
        }
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1])
        return sorted_entities[:n]

    def generate_depth_query(self, base_query: str) -> List[str]:
        """
        Generate n deep expansion queries for a base query.
        """
        n = self.config.expansion_depth
        bottom_labels = [label for label, _ in self.get_bottom_labels()]
        labels_str = ", ".join(bottom_labels)

        prompt = self.depth_prompt.format(n=n, bottom_labels=labels_str)
        messages = [
            {"role": "system", "content": "You are a Singapore-focused query expansion assistant that creates google search queries for research and knowledge base expansion"},
            {"role": "user", "content": f"{prompt}\nOriginal Query: \"{base_query}\""}
        ]

        resp = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
        )

        return self._parse_queries(resp.choices[0].message.content)

    def generate_width_query(self, base_query: str) -> List[str]:
        """
        Generate n complementary expansion queries for a base query.
        """
        n = self.config.expansion_width
        bottom_entities = [entity for entity, _ in self.get_bottom_entities()]
        entities_str = ", ".join(bottom_entities)
        prompt = self.width_prompt.format(n=n, bottom_entities=entities_str)
        messages = [
            {"role": "system", "content": "You are a Singapore-focused query expansion assistant that creates google search queries for research and knowledge base expansion"},
            {"role": "user", "content": f"{prompt}\nOriginal Query: \"{base_query}\""}
        ]

        resp = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
        )

        return self._parse_queries(resp.choices[0].message.content)

    def _expand_one_entity(self, entity: str) -> Tuple[List[str], List[str]]:
        depth_qs = self.generate_depth_query(entity)
        width_qs = self.generate_width_query(entity)
        return depth_qs, width_qs


    def auto_expand_entities(self) -> Dict[str, Dict[str, List[str]]]:
        bottoms = self.get_bottom_entities()
        if not bottoms:
            logger.warning("No bottom entities found for auto expansion")
            return {}

        expansions: Dict[str, Dict[str, List[str]]] = {}
        # You can tune max_workers or pull from config:
        max_workers = 4

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # schedule one future per entity
            future_to_entity = {
                executor.submit(self._expand_one_entity, entity): entity
                for entity, _ in bottoms
            }

            for future in as_completed(future_to_entity):
                entity = future_to_entity[future]
                try:
                    depth_qs, width_qs = future.result()
                    expansions[entity] = {
                        "depth": depth_qs,
                        "width": width_qs
                    }
                except Exception as exc:
                    logger.error(f"Expansion failed for '{entity}': {exc}")
                    expansions[entity] = {"depth": [], "width": []}

        return expansions

    def auto_expand_labels(self) -> Dict[str, List[str]]:
        """
        Automatically generate width expansions for bottom labels.
        Fallback to empty dict if no bottom labels are available.
        """
        bottoms = self.get_bottom_labels()
        if not bottoms:
            logger.warning("No bottom labels found for auto label expansion")
            return {}

        expansions: Dict[str, List[str]] = {}
        for label, _ in bottoms:
            width_qs = self.generate_width_query(label)
            expansions[label] = width_qs
        return expansions
    
    def auto_expand_all(self) -> Dict[str, Dict]:
        tasks = {
            "entities": self.auto_expand_entities,
            "labels": self.auto_expand_labels,
        }
        results: Dict[str, Dict] = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_key = {
                executor.submit(func): key for key, func in tasks.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"auto_expand_{key} failed: {e}")
                    results[key] = {}

        return results
    
    def flatten_all_expansions(self, all_expansions: Dict[str, Dict], dedupe: bool = True) -> List[str]:

        flat: List[str] = []

        for parts in all_expansions.get("entities", {}).values():
            flat.extend(parts.get("depth", []))
            flat.extend(parts.get("width", []))

        for queries in all_expansions.get("labels", {}).values():
            flat.extend(queries)

        if dedupe:
            seen = set()
            unique = []
            for q in flat:
                if q not in seen:
                    seen.add(q)
                    unique.append(q)
            return unique

        return flat


    def get_new_queries(self):
        all = self.auto_expand_all()
        print(all)
        return self.flatten_all_expansions(all)


def main():
    config = QueryExpansionConfig()
    qe = QueryExpansion(config)

    print(qe.get_new_queries())

if __name__ == "__main__":
    main()