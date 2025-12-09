import os
import json
import argparse
import sys
from typing import List, Dict

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Error: neo4j driver not installed. Please run 'pip install neo4j'")
    sys.exit(1)

def read_env(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"): continue
                if "=" in s:
                    k, v = s.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k.strip(), v)

def ingest_neo4j(triples: List[Dict], uri: str, user: str, password: str):
    print(f"Connecting to Neo4j at {uri}...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("Connected.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    count = 0
    with driver.session() as session:
        for t in triples:
            s_name = t.get("subject", "").strip()
            o_name = t.get("object", "").strip()
            pred = t.get("predicate", "").strip()
            evi = t.get("evidence", "")
            conf = t.get("confidence", 0.9)
            sec = t.get("section", "")

            if not s_name or not o_name or not pred:
                continue
            
            # Sanitize predicate slightly but allow Chinese characters.
            # We will use backticks in Cypher to handle special characters/Chinese.
            # Just ensure no backticks in the predicate itself to avoid injection.
            pred_safe = pred.replace("`", "")
            
            cypher = f"""
            MERGE (s:Entity {{name: $subj}})
            MERGE (o:Entity {{name: $obj}})
            MERGE (s)-[r:`{pred_safe}`]->(o)
            SET r.evidence = $evi, r.confidence = $conf, r.section = $sec
            """
            
            try:
                session.run(cypher, subj=s_name, obj=o_name, evi=evi, conf=conf, sec=sec)
                count += 1
                if count % 10 == 0:
                    print(f"Ingested {count} triples...", end="\r")
            except Exception as e:
                print(f"\nError ingesting {s_name} -[{pred}]-> {o_name}: {e}")

    print(f"\nIngestion completed. Total triples processed: {count}")
    driver.close()

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(root, ".env")
    read_env(env_path)

    parser = argparse.ArgumentParser(description="Ingest triples into Neo4j")
    parser.add_argument("--file", type=str, default=None, help="Path to JSON file with triples")
    args = parser.parse_args()

    # Default file path
    json_path = args.file
    if not json_path:
        json_path = os.path.join(root, "outputs_triples", "lung_cancer_triples_zh.json")
    
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        print("Please run extract_triples_zh.py first.")
        return

    # Load Triples
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    triples = data.get("triples", [])
    print(f"Loaded {len(triples)} triples from {json_path}")

    # Neo4j Config
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    neo4j_pass = os.environ.get("NEO4J_PASSWORD")

    if not (neo4j_uri and neo4j_user and neo4j_pass):
        print("Error: Neo4j credentials missing in .env (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)")
        return

    ingest_neo4j(triples, neo4j_uri, neo4j_user, neo4j_pass)

if __name__ == "__main__":
    main()
