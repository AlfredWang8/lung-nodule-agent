import os
import sys

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Error: neo4j driver not installed.")
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

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    read_env(os.path.join(root, ".env"))

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    pwd = os.environ.get("NEO4J_PASSWORD")

    if not (uri and user and pwd):
        print("Credentials missing.")
        return

    print(f"Connecting to {uri}...")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    
    print("Clearing entire database (MATCH (n) DETACH DELETE n)...")
    try:
        driver.execute_query("MATCH (n) DETACH DELETE n")
        print("Database cleared successfully.")
    except Exception as e:
        print(f"Error clearing database: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
